"""
Quick Populate Script for Nano-GraphRAG

Simple interface to quickly populate nano-graphrag storage files from documents.
"""

import sys
from pathlib import Path
from manual_nanographrag_populator import create_manual_populator


def populate_from_file(file_path: str, storage_dir: str, doc_id: str = None):
    """Populate nano-graphrag storage from a single file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Create populator
    populator = create_manual_populator(storage_dir)
    populator.load_existing_files()
    
    # Generate doc ID if not provided
    if not doc_id:
        doc_id = f"doc-{populator.generate_id(file_path.name)}"
    
    # Process document
    print(f"Processing {file_path.name}...")
    populator.process_document_auto(doc_id, content, file_path.stem)
    
    # Save files
    populator.save_all_files()
    
    print(f"Successfully populated storage with:")
    print(f"- {len(populator.entities)} entities")
    print(f"- {len(populator.relationships)} relationships")
    print(f"- {len(populator.text_chunks)} text chunks")
    print(f"- Storage saved to: {storage_dir}")


def populate_from_folder(folder_path: str, storage_dir: str):
    """Populate nano-graphrag storage from all files in a folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Create populator
    populator = create_manual_populator(storage_dir)
    populator.load_existing_files()
    
    # Process all text files
    text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml'}
    processed_files = 0
    
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc_id = f"doc-{populator.generate_id(str(file_path))}"
                print(f"Processing {file_path.name}...")
                populator.process_document_auto(doc_id, content, file_path.stem)
                processed_files += 1
                
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
    
    # Save files
    populator.save_all_files()
    
    print(f"Successfully processed {processed_files} files:")
    print(f"- {len(populator.entities)} total entities")
    print(f"- {len(populator.relationships)} total relationships")
    print(f"- {len(populator.text_chunks)} total text chunks")
    print(f"- Storage saved to: {storage_dir}")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python quick_populate.py file <file_path> <storage_dir> [doc_id]")
        print("  python quick_populate.py folder <folder_path> <storage_dir>")
        print()
        print("Examples:")
        print("  python quick_populate.py file paper.pdf ./nanographrag_storage/chat_1/session_1")
        print("  python quick_populate.py folder ./papers ./nanographrag_storage/chat_1/session_1")
        return
    
    command = sys.argv[1]
    
    if command == "file":
        if len(sys.argv) < 4:
            print("Error: file command requires <file_path> <storage_dir> [doc_id]")
            return
        
        file_path = sys.argv[2]
        storage_dir = sys.argv[3]
        doc_id = sys.argv[4] if len(sys.argv) > 4 else None
        
        populate_from_file(file_path, storage_dir, doc_id)
        
    elif command == "folder":
        if len(sys.argv) < 4:
            print("Error: folder command requires <folder_path> <storage_dir>")
            return
        
        folder_path = sys.argv[2]
        storage_dir = sys.argv[3]
        
        populate_from_folder(folder_path, storage_dir)
        
    else:
        print(f"Error: Unknown command '{command}'. Use 'file' or 'folder'.")


if __name__ == "__main__":
    main()