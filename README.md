# minGrok

This repo is meant as a guide to how XAI's newly open-sourced model Grok-1 works. To see their original implementation, [click here](https://github.com/xai-org/grok-1). To find the googel colab notebook that walks through the architecture in *excruciating* detail as a demonstration for beginners, [click here](https://colab.research.google.com/drive/1o3RV23gIDVcfkxgTe2jnbLTKVyYuZTMM?usp=sharing) and check out my youtube video where I walk through it below. If you're not a beginner (already knowledgeable about decoder-only transformers) then I recommend skimming through `model.py` and `config.py` to see all the ways in which Grok-1 differs from other open-source models like Llama, Mistral and Gemma.

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/K9Rdc848EBs/0.jpg)](https://www.youtube.com/watch?v=K9Rdc848EBs)

### Repo Contents

- [The Accompanying Colab Notebook](https://colab.research.google.com/drive/1o3RV23gIDVcfkxgTe2jnbLTKVyYuZTMM?usp=sharing) - the teaching material I walk through in [my youtube video]()
- `minGrok_train-test.ipynb` - the notebook where I actually trained the 1m parameter model. The code here is essentially the same as what's in section 3 of the colab notebook
- `model.py` - contains the nn.Modules used to define minGrok. The code here is essentially the same as what's in section 2 of the colab notebook
- `config.py` - contains minGrok's configuration hyperparameters as well as comments indicating what full-sized Grok uses
- `tokenizer.py` - a very simple tokenizer of length 128 built off of TinyShakespeare's original 65 character vocabulary. By no means should anyone actually use this in production but it's fine as a simple stand-in given that the purpose of this repo is not to teach about tokenization
- `input.txt` - just [TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). If i wasn't so lazy I would've set all this code to download it directly rather than actually storing a copy in this repo
- `models/` - a folder of 1m parameter model(s) that i trained on my macbook air. Again don't expect anything impressive, they're just here for teaching purposes so that you can load them rather than training your own. If you train something bigger feel free to upload I guess, but stick with my lazy practice of designating hyperparameters in the title

### ToDo
- [ ] A commenter pointed out my lack of inclusion of MoE specific training dynamics. Basically in order to encourage proper expert utilization rather than over-reliance on one expert, you need to both add randomness to the Router's logits and add a diversity loss to ensure every expert is used in every batch. The video will not be changing but fingers crossed I'll be able update the code today. Should be a good little test if you've actually read the code.

### Check out my socials
- [Youtube](https://www.youtube.com/channel/UCeQhm8DwHBg_YEYY0KGM1GQ)
- [Linkedin](https://tr.ee/HgIcstKnBX)
- [my Discord server](https://tr.ee/WwukUOvWIc)
- [Patreon](https://tr.ee/UH_v1ThFD1)
- [Weekly Newsletter](https://tr.ee/hIEnMCPQaI)
