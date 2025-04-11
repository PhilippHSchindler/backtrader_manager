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

def is_iterable(obj):
    if isinstance(obj, (str, bytes)):  # Exclude strings and bytes
        return False
    try:
        iterator = iter(obj)  # Check if obj can be iterated over
        first_item = next(iterator)  # Try getting the first item
        second_item = next(iterator, None)  # Try getting a second item
        return second_item is not None  # If there's no second item, return False
    except TypeError:
        return False
    except StopIteration:  
        return False  # If the iterable is empty, return False

def _format_parameters(parameters):
    parameters_formatted = {}
    
    if parameters==None or (parameters=={}):
        return {}
        
    for key, parameter in parameters.items():
        key_lower = key.lower()
    
        # Handle different types of parameter values
        if isinstance(parameter, (list, tuple, set)):
            parameters_formatted[key_lower]= parameter  # Ensure it's a list
        else:
            parameters_formatted[key_lower] = [parameter]  # Wrap single value in a list
    
    return parameters_formatted