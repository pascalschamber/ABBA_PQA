import time

class TimeRecordingClass:
    def __init__(self):
        self.execution_times = {}

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)

        # Check if the attribute is a method
        if callable(attr):
            def timed(*args, **kwargs):
                start_time = time.time()
                result = attr(*args, **kwargs)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                
                # Store the execution time
                if name not in self.execution_times:
                    self.execution_times[name] = []
                self.execution_times[name].append(elapsed_time)

                return result
            return timed

        return attr

    # Example methods
    def run(self):
        # Simulate a task
        time.sleep(1)

    def another_method(self):
        # Another simulated task
        time.sleep(0.5)

# Example usage
obj = TimeRecordingClass()
obj.run()
obj.another_method()
print(obj.execution_times)
