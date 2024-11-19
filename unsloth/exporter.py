from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import tempfile
import os
import json
class ModelExporter:
    """
    Base class for exporting models.
    """
    def __init__(self):
        pass
    
    
class LVMMExporter(ModelExporter):
    """
    Export models to vLLM format.
    """
    def __init__(self):
        pass

    @staticmethod
    def export_peft(output_dir: str, model: AutoModel, tokenizer: AutoTokenizer, base_model: str=None, architecture_name: str=None):
        """
        Export a PEFT (LoRa) model to vLLM format.
        
        Args:
            output_dir (str): The directory to save the exported model.
            model (AutoModel): The PEFT model to export.
            tokenizer (AutoTokenizer): The tokenizer for the model.
            base_model (str): The base model to use for the export, can be HF or local path.
            architecture_name (str): The architecture name for the vLLM config, should be supported by vLLM https://docs.vllm.ai/en/latest/models/supported_models.html
        """
        
        if not isinstance(model, AutoModel):
            raise ValueError("Model must be an AutoModel")
        
        if not isinstance(model, PeftModel):
            raise ValueError("Model must be a PeftModel")
        
        if not isinstance(tokenizer, AutoTokenizer):
            raise ValueError("Tokenizer must be an AutoTokenizer")
        
        # get the name of the base model from the PEFT model
        if not base_model:
            base_model = model.name_or_path
            print(f'Using base model : {base_model}')
            
        # get architecture name for vllm config 
        if not architecture_name:
            architecture_name = next(model.model.modules()).__class__.__name__
            print(f'Using architecture : {architecture_name}')
        
        # save the lora adapter
        if not os.path.exists(output_dir):
            print(f'Creating output directory : {output_dir}')
            os.makedirs(output_dir)

        save_model_path = os.path.join(output_dir, 'lora_adapter')
        model.save_pretrained(save_model_path)
        
        save_tokenizer_path = os.path.join(output_dir, 'tokenizer')
        tokenizer.save_pretrained(save_tokenizer_path)
        
        # save the vllm config
        vllm_config_path = os.path.join(output_dir, 'vllm_config.json')
        with open(vllm_config_path, 'w') as f:
            json.dump({
                'base_model': base_model,
                'architecture_name': architecture_name
            }, f)
        
        run_command = (
            f"vllm serve {base_model} "
            f"--enable-lora "
            f"--lora-modules custom-lora={output_dir}"
        )
        print(run_command)