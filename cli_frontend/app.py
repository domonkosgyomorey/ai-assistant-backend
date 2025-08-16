import time

from colorama import Fore, just_fix_windows_console
from common_types import COMMANDS
from communication import Communication


class Frontend:
    def __init__(self, communication: Communication):
        self.communication = communication

    def run(self) -> None:
        print(
            f"{Fore.CYAN}🚀 AI Assistant CLI Frontend{Fore.RESET}\n"
            "Welcome to the AI Assistant! Ask me anything.\n"
            "\n"
            f"{Fore.CYAN}Available Commands:{Fore.RESET}\n"
            f"\t{Fore.GREEN}exit{Fore.RESET}: Exit the application\n"
            f"\t{Fore.GREEN}save{Fore.RESET}: Save conversation history to file\n"
            f"\t{Fore.YELLOW}clear{Fore.RESET}: Clear conversation history\n"
            "\n"
        )

        run: bool = True
        while run:
            try:
                message = input(f"[{Fore.CYAN}You{Fore.RESET}]: ").strip()
                print()

                if not message:
                    print(f"{Fore.YELLOW}⚠️ Please enter a message or command.{Fore.RESET}\n")
                    continue

                if message.lower() == COMMANDS.EXIT:
                    print(f"{Fore.GREEN}👋 Goodbye! Thanks for using AI Assistant.{Fore.RESET}")
                    run = False
                elif message.lower() == COMMANDS.SAVE_CONV:
                    print(f"{Fore.BLUE}💾 Saving conversation...{Fore.RESET}")
                    self.communication.save_conversion()
                    print(f"{Fore.GREEN}✅ Conversation saved successfully!{Fore.RESET}\n")
                elif message.lower() == "clear":
                    self.communication.conversations = {"messages": []}
                    print(f"{Fore.GREEN}🗑️ Conversation history cleared.{Fore.RESET}\n")
                else:
                    print(f"{Fore.BLUE}⏱️ Processing your request...{Fore.RESET}")
                    start_time = time.time()

                    self.communication.send_message(message)

                    elapsed_time = time.time() - start_time
                    print(f"{Fore.BLUE}⌛ Total response time: {Fore.YELLOW}{elapsed_time:.2f}s{Fore.RESET}\n")
                    print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}🔄 Use 'exit' to quit or continue chatting...{Fore.RESET}\n")
            except Exception as e:
                print(f"{Fore.RED}❌ An error occurred: {str(e)}{Fore.RESET}\n")


if __name__ == "__main__":
    just_fix_windows_console()
    Frontend(Communication()).run()
