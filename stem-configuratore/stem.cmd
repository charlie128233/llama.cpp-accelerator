@echo off
rem stem: configuratore di llama.cpp per questo PC (vedi README.md). Esempio: stem init -m models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf
python "%~dp0tools\stem\stem.py" %*
