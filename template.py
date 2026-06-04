import os
import pathlib

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def create_insurance_rag_structure():
    """Create the insurance RAG project structure with empty files"""
    
    # Define the complete folder structure
    structure = {
        "": [  # Root directory files
            "app.py",
            "requirements.txt",
            ".env",
            "config.yaml",
            "exp_nb.ipynb",
            "readme.md"
        ],
        "data/raw/insurance_docs": [
            "policy_terms.txt",
            "claim_procedure.txt"
        ],
        "data/embeddings": [
            "faiss_index/"
        ],
        "utils": [
            "loader.py",
            "retriever.py",
            "query_rewriter.py",
            "hyde_generator.py"
        ]
    }
    
    print("Creating Insurance RAG Project structure...")
    
    # Create all folders and files
    for folder, items in structure.items():
        # Create folder if it doesn't exist
        if folder:
            os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)
            print(f"📁 Created folder: {folder}/")
        
        # Create files within the folder
        for item in items:
            file_path = os.path.join(ROOT_DIR, folder, item) if folder else os.path.join(ROOT_DIR, item)
            
            # Handle directories (ending with /)
            if item.endswith('/'):
                os.makedirs(file_path, exist_ok=True)
                print(f"📁 Created folder: {file_path}")
            else:
                # Create empty file
                pathlib.Path(file_path).touch()
                print(f"📄 Created file: {file_path}")
    
    print("\n✅ Insurance RAG Project structure created successfully!")
    print("📂 All files and folders are now ready.")

if __name__ == "__main__":
    create_insurance_rag_structure()