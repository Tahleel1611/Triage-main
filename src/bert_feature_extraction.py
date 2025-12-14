import pandas as pd
import numpy as np
import simple_icd_10 as icd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def get_description(code):
    """Convert ICD-10 code to description."""
    code = code.strip()
    if not code:
        return ""
        
    # Try raw
    if icd.is_valid_item(code):
        return icd.get_description(code)
    
    # Try adding dot after 3rd char
    if len(code) > 3:
        code_dot = code[:3] + "." + code[3:]
        if icd.is_valid_item(code_dot):
            return icd.get_description(code_dot)
            
    return code  # Return code itself if no description found

def process_chief_complaint(cc_str):
    """Convert string of codes to string of descriptions."""
    if pd.isna(cc_str):
        return "Unknown"
    
    codes = str(cc_str).split()
    descriptions = [get_description(c) for c in codes]
    return " ".join(descriptions)

def main():
    data_path = 'data/nhamcs_combined.csv'
    output_path = 'data/nhamcs_bert_features.npy'
    text_output_path = 'data/nhamcs_text_descriptions.csv'
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 1. Convert Codes to Text
    print("Mapping ICD-10 codes to text descriptions...")
    tqdm.pandas()
    df['Chief_complain_text'] = df['Chief_complain'].progress_apply(process_chief_complaint)
    
    # Save text descriptions for inspection
    df[['Chief_complain', 'Chief_complain_text']].to_csv(text_output_path, index=False)
    print(f"Text descriptions saved to {text_output_path}")
    
    # 2. Load ClinicalBERT
    print("Loading ClinicalBERT model...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # 3. Extract Embeddings
    print("Extracting BERT embeddings...")
    batch_size = 32
    texts = df['Chief_complain_text'].tolist()
    
    # For testing/demo purposes, uncomment the next line to limit to 1000 rows
    texts = texts[:1000] 
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    final_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {final_embeddings.shape}")
    
    # 4. Save
    np.save(output_path, final_embeddings)
    print(f"BERT embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
