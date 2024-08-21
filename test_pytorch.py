
import torch
import sys
import os


def check1():
    print(os.environ.get('CUDA_HOME', 'CUDA_HOME is not set'))

def main():
    print("hello world")
    # Print Python version
    print(f"Python version: {sys.version}")
    
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available and print CUDA version
    if torch.cuda.is_available():
        print(f"CUDA is available. CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available.")

    print(torch.utils.cpp_extension.CUDA_HOME)

    
    
    # # Generate and print a random number using PyTorch
    # random_number = torch.rand(1).item()
    # print(f"Generated random number: {random_number}")

if __name__ == "__main__":
    main()
    # check1()




# import torch
# import transformers
# import llava  # Replace with the actual module name if different

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Transformers version:", transformers.__version__)

