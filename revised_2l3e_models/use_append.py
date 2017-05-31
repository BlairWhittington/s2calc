import append_LR_s2
from append_LR_s2 import *

def main():
	append = append_s2()
	append.load_set()
	append.load_saved_model()
	append.append_s2()
	
if __name__ == "__main__":
	main()
