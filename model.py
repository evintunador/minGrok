#from config import *
import torch
import torch.nn as nn 
import torch.nn.functional as F  

def apply_rotary_emb(x: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Applies the rotary embedding to the inputted query or key tensor"""
    # Get sequence length
    seq_len = x.size(1)
    device = x.device

    # Dynamically compute frequency cis based on the input sequence length
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    # Apply rotary embeddings to the input tensor
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis.unsqueeze(0)).type_as(x)  # Ensure batch dimension is handled
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)

    return x_out

class MQA(nn.Module):
    """
    Implements Multi-Query Attention which supports a distinct number of attention heads for queries and key-values (KV).
    In the case where the same number of queries and key-values are used, this implemenation is equivalent to regular Multi-Head Attention.
    """
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.theta = config.rope_theta

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Create a mask tensor with shape [batch_size, num_heads, seq_len, seq_len]
        self.mask = torch.tril(torch.ones((config.max_position_embeddings, config.max_position_embeddings), 
                                     dtype=torch.uint8)).view(1, 1, config.max_position_embeddings, config.max_position_embeddings).to(dtype=torch.bool)
        #self.mask = mask.expand(-1, self.num_heads, -1, -1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        batch_size, input_len, _ = hidden_states_shape

        # Applies the linear projection to the hidden state to retrieve our q, k & v projections
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],dim=-1)

        # Reshapes each to separate the heads and align the dimensions for attention operations.
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Applies rotary positional embeddings to queries and keys to incorporate positional information.
        xq = apply_rotary_emb(xq, self.head_dim, self.theta)
        xk = apply_rotary_emb(xk, self.head_dim, self.theta)

        # If the number of KV heads is different from the number of query heads, adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_heads:
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)

        # Transposes to align them for the batch matrix multiplication in attention calculation.
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Calculates attention logits by performing a batch matrix multiplication between queries and keys
        logits = torch.matmul(q, k.transpose(2, 3))

        # Grok's unusual scaling method
        # If anyone knows why they use 0.08838834764831845 in Grok please lmk. Maybe it's a learned value?
        logits *= 0.08838834764831845
        # Next here we'll scale and clip our attention logits
        # the tanh is a nonlinear function that pushes all of the entries in logits into the range (-1, 1)
        # then they're scaled up to the range (-30, 30). The number 30 is an arbitrary choice
        # the purpose of this scaling is to regularize and prevent numerical stability that might otherwise mess with the upcoming softmax
        max_attn_val = torch.tensor(30.0, dtype = logits.dtype)
        logits = max_attn_val * torch.tanh(logits / max_attn_val)
        # other transformers would replace the last three lines with a multiplication by torch.sqrt(self.hidden_size)

        # Applies the lower-triangular mask to the attention logits
        logits = torch.where(self.mask[..., :input_len, :input_len].expand_as(logits), logits, torch.tensor(-1e30, device=logits.device, dtype=logits.dtype))

        # Applies softmax to the logits to obtain attention probabilities
        scores = F.softmax(logits, dim=-1)

        # Computes the weighted sum of values based on the attention scores to obtain the output of the attention mechanism.
        output = torch.matmul(scores, v)

        # Reshapes the attention output to match the expected output dimensions, combining the heads back into the hidden dimension.
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)

        # Applies the final linear projection to the attention output, mapping it back to the hidden size dimension.
        output = self.o_proj(output)

        return output

class Expert(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(model_dim, hidden_dim * 2, bias=False)  # Double the output for gating
        self.layer2 = nn.Linear(hidden_dim, model_dim, bias=False)  # Output layer remains the same

    def forward(self, x):
      # Split the output of the first layer for gating
        x, gate = self.layer1(x).chunk(2, dim=-1)

        # Apply GeLU to the gate, and then multiply element-wise
        x = F.gelu(gate) * x
        x = self.layer2(x)

        return x

class Router(nn.Module):
    def __init__(self, input_size, tot_num_experts, noise_std: float = 0.1):
        super().__init__()
        self.tot_num_experts = tot_num_experts
        self.router_weights = nn.Linear(input_size, tot_num_experts, bias=False)
        self.noise_std = noise_std

    def forward(self, inputs, training: bool = False):
        routing_logits = self.router_weights(inputs)
        if training: routing_logits = routing_logits + torch.randn_like(routing_logits) * self.noise_std
        routing_probs = F.softmax(routing_logits, dim=-1)
        return routing_probs

class MoELayer(nn.Module):
    def __init__(self, model_dim, expert_hidden_dim, tot_num_experts, chosen_num_experts, noise_std):
        super().__init__()
        self.model_dim = model_dim
        self.tot_num_experts = tot_num_experts
        self.chosen_num_experts = chosen_num_experts
        self.experts = nn.ModuleList([Expert(model_dim, expert_hidden_dim) for _ in range(tot_num_experts)])
        self.router = Router(model_dim, tot_num_experts, noise_std)

    def forward(self, inputs, training: bool = False):
        b, seq_len, _ = inputs.shape

        # get the output of all the experts
        expert_outputs = [expert(inputs.view(-1, self.model_dim)) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=0).view(b, seq_len, self.tot_num_experts, self.model_dim)

        # get the output of the router and create out expert mask
        routing_probs = F.softmax(self.router(inputs), dim=-1)
        with torch.no_grad():
          expert_indices = torch.topk(routing_probs, k=self.chosen_num_experts, sorted=True).indices
          multi_hot_indices = torch.zeros(b, seq_len, self.tot_num_experts, device=inputs.device)
          multi_hot_indices = multi_hot_indices.scatter(2, expert_indices, 1)

        # Apply the multi-hot mask (first expand dimensions for broadcasting)
        multi_hot_expanded = multi_hot_indices.unsqueeze(-1).expand_as(expert_outputs)
        output_masked = expert_outputs * multi_hot_expanded.float()

        # then weight our experts' outputs by the softmax values (which we first must broadcast to the right shape) and sum them
        routing_probs_expanded = routing_probs.unsqueeze(-1).expand_as(output_masked)
        MoE_output = (output_masked * routing_probs_expanded).sum(dim=2)

        return MoE_output, routing_probs # we also output routing_probs to be used in the loss function later

class RMSNorm(nn.Module): # the same RMSNorm we wrote earlier
    def __init__(self, num_features, eps=1e-5, use_scale=True):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features)) if use_scale else None

    def forward(self, inputs):
        # Calculate the mean squared value for each feature
        mean_squared = inputs.pow(2).mean(dim=-1, keepdim=True)

        # Normalize inputs
        normed_inputs = inputs * torch.rsqrt(mean_squared + self.eps)

        # Apply scale if it exists
        if self.scale is not None:
            normed_inputs = normed_inputs * self.scale

        return normed_inputs

class DecoderLayer(nn.Module):
    """
    A decoder layer that integrates the Attention mechanism and MoE. It includes
    normalization steps both before and after the MQA and MoE but never actually normalized the residual connection
    """

    def __init__(self, config):
        super().__init__()

        self.mqa = MQA(config)

        self.moe = MoELayer(
            model_dim = config.hidden_size,
            expert_hidden_dim = config.hidden_size * config.embedding_multiplier_scale,
            tot_num_experts = config.tot_num_experts,
            chosen_num_experts = config.chosen_num_experts,
            noise_std = config.noise_std
        )

        self.pre_mqa_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps, use_scale = config.use_scale)
        self.post_mqa_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps, use_scale = config.use_scale)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps, use_scale = config.use_scale)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps, use_scale = config.use_scale)

        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            x = x + self.drop(self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x))))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.drop(self.post_moe_norm(moe_out))
        else:
            x = x + self.post_mqa_norm(self.mqa(self.pre_mqa_norm(x)))
            moe_out, routing_probs = self.moe(self.pre_moe_norm(x), training)
            x = x + self.post_moe_norm(moe_out)
        return x, routing_probs

class minGrok(nn.Module):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config

        # the attention heads need to cleanly divide up the hidden_size of the model so that we can split it all apart & combine back together
        assert config.hidden_size % config.num_attention_heads == 0

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer

         # the embedding matrix. for converting tokens to the first residual state, and the last residual state to logits
        self.embedder = nn.Embedding(self.vocab_size, config.hidden_size)

        # Initialize a sequence of DecoderLayer instances as specified by the number of layers in the config
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))

        # Initialize a normalization layer to be applied after the last decoder layer, stabilizing the output
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # the primary loss function
        self.criterion = nn.CrossEntropyLoss()
    
        # the hyperparameter weighting the secondary loss function
        self.lambadada = config.lambadada

    def calc_moe_loss(self, routing_probs_list):
        # this is silly and inefficient but i'm tired and bored of this project ngl
        # basically i'm choosing to sum the per-layer MoE variances
        cum_var = torch.tensor(0.0) # this will be encouraged to be 0 so it doesn't even matter if we record the gradient
        for routing_probs in routing_probs_list:
            expert_usage = routing_probs.sum(dim=0)
            usage_mean = expert_usage.mean()
            expert_variance = ((expert_usage - usage_mean) ** 2).mean()
            cum_var = cum_var + expert_variance

        return cum_var

    # a more efficient version ChatGPT4 made that i'm too lazy to test, but go ahead if you want
    #def calc_moe_loss(self, routing_probs_list):
        # Concatenate all tensors along a new dimension (say, dim=0)
        # This results in a new tensor of shape (N, b, t, c) where N is the number of tensors in routing_probs_list
        #all_routing_probs = torch.cat([x.unsqueeze(0) for x in routing_probs_list], dim=0)
        
        # Sum across the batch (b) and time (t) dimensions, resulting in a shape of (N, c)
        #expert_usage = all_routing_probs.sum(dim=1).sum(dim=1)
        
        # Calculate the mean across the new dimension (N) and the experts (c), resulting in a single mean value
        #usage_mean = expert_usage.mean(dim=0).mean(dim=0)
        
        # Calculate the variance
        #expert_variance = ((expert_usage - usage_mean) ** 2).mean(dim=0).mean(dim=0)
        
        # Sum the variance across all layers (N)
        #cum_var = expert_variance.sum()
        
        #return cum_var

    def forward(
        self,
        input_token_ids: torch.Tensor, # a shape (batch_size, input_seq_len) list of integer token ids
        target_token_ids: torch.Tensor = None, # a shape (batch_size, input_seq_len) list of token ids to train on
        ) -> torch.Tensor:
        training = False if target_token_ids is None else True

        # turn the input tokens into the first resudial state using the embedding matrix
        x = self.embedder(input_token_ids) * self.config.hidden_size**0.5 # Grok normalizes the embedding by sqrt(hidden_size)

        # initialize a list to store the routing probs of each layer in
        routing_probs_list = []
        # Iteratively process the input through each DecoderLayer
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, routing_probs = layer(x, training)
            if training: routing_probs_list.append(routing_probs)

        # Apply normalization to the output of the final decoder layer
        x = self.final_norm(x)

        # grabbing the weights of the embedding matrix shape (vocab_size, hidden_dim) for use as the output layer
        embedder_weight = self.embedder.weight

        # the embedding matrix is also used as the output layer
        # this saves on parameters & makes sense for interpretability
        # (batch_size, input_len, hidden_size) @ (hidden_size, vocab_size) -> (batch_size, input_len, vocab_size)
        logits = torch.matmul(x, embedder_weight.t())

        if training: # if we are training
            batch_size, input_len, vocab_size = logits.shape

            # we reshape our logits & targets before calculating cross-entropy loss
            CEloss = self.criterion(logits.view(batch_size*input_len, vocab_size),
                                    target_token_ids.view(batch_size*input_len))
            
            # calculating the MoE loss that encourages all experts to be utilized
            MoEloss = self.calc_moe_loss(routing_probs_list)

            # our final loss value
            loss = CEloss + MoEloss * self.lambadada
        else:
            loss = None # if we're not training, then we don't need to calculate loss

        return logits, loss

    @torch.no_grad() # no need to keep track of gradients during inference
    def Sampler(
        self,
        logits: torch.Tensor, # shape (batch_size, input_len, vocab_size)
        temperature: float, # controls how boring vs random the outputs should be
        top_p: float, # the maximum cumulative probability of output options we're willing to consider
        top_k: int, # the maximum number of output options we're willing to consider
    ) -> torch.Tensor:
        """
        The Sampler function is responsible for generating token predictions from Grok's output.
        It supports temperature scaling, top-p (nucleus) sampling, and top-k sampling
        """
        # Select the last element for each sequence.
        logits = logits[:,-1,:]

        # Apply temperature scaling
        logits.div_(temperature) # div_ is an in-place operation which is ok since we don't record gradients during inference

        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # sort the probabilities to for use in top-p & top-k
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # calculating top_k
        probs_sum = torch.cumsum(probs_sort, dim=-1) # creates same-size tensor of cumulatve probabilities instead of indivdiual probs
        top_ps_mask = (probs_sum - probs_sort) > top_p # mask where 0's are top-p selections & 1's are to be excluded
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)  # the original probabilities with excluded tokens changed to 0.0

        # calculating top_k
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device) # create a shape (vocab_size) tensor that just iterates up by 1's
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1) # expand our mask along the batch_size dimension to become size (batch_size, vocab_size)
        top_ks_mask = top_ks_mask >= top_k # top_ks is a list of integers. we keep whichever entries in top_ks_mask are greater than their corresponding entries in top_ks

        # we'll be combining top-p with top-k and using whichever gives us fewer tokens. a very conservative approach
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # Re-normalization so that total probabilities add up to 1
        # now we rearrange the modified probabilities in probs_sort back to their original order according to probs_idx
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        # samples from the distribution
        next_token_id = torch.multinomial(probs, num_samples=1)

        return next_token_id

    def generate(
        self,
        prompt: str,
        output_len: int = 100, # the model will output 100 tokens
        temperature: float = 0.95, # 0.95 is pretty close to not even using temperature at all (1.0 would be no effect)
        top_p: float = 1.0, # defaulting to 1 means we essentially don't use top-p
        top_k: int = 65, # setting top_k = vocab_size means we're effectively not using top_k at all
    ) -> str:
        """Generates responses for given prompts using Grok model."""

        # encoding the prompt into token indices
        tokens = self.tokenizer.encode(prompt)

        # turning it into the right tensor shape
        tokens = torch.tensor(tokens, device=self.config.device).unsqueeze(0)

        # we wouldn't want to go past the maximum context length we trained on
        assert len(tokens) + output_len <= self.config.max_position_embeddings

        for i in range(output_len):
            # get the model's output logits and ignore the loss, which would be a NoneType object
            logits, _ = self(tokens[:,:self.max_seq_len])

            next_token = self.Sampler(
                logits = logits, # the actual output of the model
                temperature = temperature,
                top_p = top_p,
                top_k = top_k
            )

            # add our new token to the sequence
            tokens = torch.cat((tokens, next_token), dim=1)

        # decode our list of tokens to an actual string
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())

        return output