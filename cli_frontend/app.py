import time

from colorama import Fore, just_fix_windows_console
from common_types import COMMANDS
from communication import Communication
from utils import num_of_bytes


class Frontend:
    def __init__(self, communication: Communication):
        self.communication = communication

    def run(self) -> None:
        print(
            "This is a basic cli frontend.\n"
            "\n"
            f"{Fore.CYAN}Commands:{Fore.RESET}\n"
            f"\t{Fore.GREEN}exit{Fore.RESET}: Exits from the frontend.\n"
            f"\t{Fore.GREEN}save{Fore.RESET}: Saves the previous conversation into a json file.\n"
        )

        message: str = ""
        run: bool = True
        while run:
            message = input(f"[{Fore.CYAN}User{Fore.RESET}]: ")
            print()
            if message.strip().lower() == COMMANDS.EXIT:
                run = False
            elif message.strip().lower() == COMMANDS.SAVE_CONV:
                self.communication.save_conversion()
            else:
                start = time.time()
                response: str = self.communication.send_message(message)
                response_time = round(time.time() - start, 3)
                initial_msg = f"[{Fore.GREEN}AI{Fore.RESET}] - response {Fore.RED}time{Fore.RESET} {response_time}s\n"
                comm_bytes = f"{Fore.RED}bytes in{Fore.RESET}:[{num_of_bytes(message)}] {Fore.RED}out{Fore.RESET}:[{num_of_bytes(response)}]\n"
                print(initial_msg + comm_bytes + response + Fore.RESET + "\n")


if __name__ == "__main__":
    just_fix_windows_console()
    Frontend(Communication()).run()
