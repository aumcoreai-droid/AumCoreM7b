
def start_console():
    print("🚀 AumCoreM7b Interactive Debug Console")
    print("Commands: debug, history, exit")

    while True:
        command = input(">>> ").strip().lower()
        if command == "exit":
            break
        elif command == "debug":
            print("Debug mode activated...")
        elif command == "history":
            print("Session history: No data yet")
        else:
            print("Unknown command")
