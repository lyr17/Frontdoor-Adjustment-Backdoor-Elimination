# FABE

We use [Tuna](https://github.com/microsoft/LMOps/tree/main/tuna) finetune an instruction-tuned LLM as FABE model, the base LLM model is LlaMA2-7B-Chat, then we use [Openbackdoor](https://github.com/thunlp/OpenBackdoor) as a framework to finish the defensive process of FABE.

## Installation

you can install FABE through Git.

**Git**

```
https://github.com/lyr17/Instruct-as-backdoor-cleaner.git
cd Instruct-as-backdoor-cleaner
pip install -r requirements.txt
```

## Usage

### Step 1 : finetune the base model to get FABE model

```
cd Tuna
bash src/train_tuna.sh data/llama_file.json 1e-5
cd ..
```

We use 8×V100 GPU train the model 24 hours to get the finetuned model and model path like *tuna/src/checkpoints/tuna_p/checkpoint-3024* .

### Step 2 : defense the victim model by using FABE model

You can configure the hyperparameters of FABE in these path ./configs/fabe_config.json, please set the hyperparameter "model_path" for fabe in this json file to the model path saved in step 1.

The hyperparameter "diversity" is the model generation diversity penalty for FABE model. When defending against BadNets and AddSent attacks, it is recommended to set diversity to 0.1, when defending against SynBkd attack, it is recommended to set diversity to 1.0.

```
cd OpenBackdoor
python FABE_defense.py --config_path ./configs/fabe_config.json
cd ..
```

