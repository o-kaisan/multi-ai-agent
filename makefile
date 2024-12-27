.PHONY deps
deps:
	sudp apt update
	sudo apt install python3-pydot graphviz
	sudo pip install -r requirements.txt