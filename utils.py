# print headlines
def print_headline(text, level = None):
    separator = '='
    if level == 2:
        separator = '-'
    if level == 3:
        separator = '.'
        
    print(f"\n{separator * len(text)}")
    print(text)
    print(f"{separator * len(text)}")

# wait for confirmation by user
def wait_for_user_confirmation():
    user_input = input("Do you want to proceed? (yes/no): ").strip().lower()
    return user_input in {"yes", "y"}

