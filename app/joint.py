from abc import ABC, abstractmethod


# Abstract class (No instance of this class can be created)
class Joint(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name


class Neck(Joint):
    def __init__(self, name: str):
        super().__init__(name)


class Shoulder(Joint):
    def __init__(self, name: str):
        super().__init__(name)


class Waist(Joint):
    def __init__(self, name: str):
        super().__init__(name)


class Ankle(Joint):
    def __init__(self, name: str):
        super().__init__(name)
