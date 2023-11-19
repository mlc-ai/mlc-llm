"""A command line tool for directly chatting with open source LLMs"""
from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout
import os
from ..support.argparse import ArgumentParser



# From the mlc-llm directory, run
# $ python mlc_chat_cli.py

def printhelp():
    print("You can use the following special commands:")
    print("\t/help\t\tprint the special commands")
    print("\t/exit\t\tquite the cli")
    print("\t/stats\t\tprint out the latest stats (token/sec)")
    print("\t/reset\t\trestart a fresh chat")
    print("\t/reload [model]\treload model `model` from disk, or reload the current model if `model` not not specified")


def chat(
    modelname: str    
    ):
    # Create a ChatModule instance
    cm = ChatModule(model=modelname, device="auto")
    # cm = ChatModule(model="Mistral-7B-Instruct-v0.1-q4f16_1")

    print("Using MLC config: ", os.path.abspath( cm.config_file_path) )
    print("Using model weights:",os.path.abspath( cm.model_path))
    print("Using model library:", os.path.abspath(cm.model_lib_path))
    printhelp()


    PS=cm._get_role_0()

    prompt=input(f"{PS} ")

    while( prompt != '/exit'):
        match prompt:
            case "/help":
                printhelp()
            case "/stats":
                print(f"Statistics: {cm.stats()}\n")
            case "/reset":
                cm.reset_chat()
            case "/reload":  
                cm._reload(os.path.abspath(cm.model_lib_path), os.path.abspath(cm.model_path), os.path.abspath(cm.config_file_path))
            case _:
                output = cm.generate(
                    prompt,
                    progress_callback=StreamToStdout(callback_interval=2),
                )
        prompt=input(f"{PS} ")

    # Print prefill and decode performance statistics
    print(f"Statistics: {cm.stats()}\n")

def main(argv):

    parser = ArgumentParser(description="Chat with open source LLMs")
    parser.add_argument(
        "--model",
        type=str,
        default="Llama-2-7b-chat-hf-q4f16_1",
        help="full name of LLM model",
        required=True,
    )
    args = parser.parse_args(argv)
    chat(
        modelname=args.model
    )

if __name__ == "__main__":
    main()
