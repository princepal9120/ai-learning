#basic class and object
# problem-> create a car class with attributes like brand and model. then create a instance of this class.
#self is keyworkd which is used for context (in js like this)
class Car:
    total_car=0
    def __init__ (self,brand,model): #parameter1 ,parameter2
        self.__brand=brand # class brand , class model where self
        self.__model=model
        Car.total_car+=1

        # access private element, encapusiation
    def get_brand(self):
        return self.__brand+" !"      
    #  mehtod and self
    def full_name(self):
        return f"{self.__brand} {self.__model}"
    
    #polymorism
    def fuel_type(self):
        return "Petrol or Diesel"
    # static method
    @staticmethod
    def general_description():      
        return "Cars are means of transportation" 
    
    @property
    def model(self):
        return self.__model

    
#Inheritance
class ElectricCar(Car):
    def __init__(self, brand, model,battery_size):
        super().__init__(brand, model)
        self.battery_size=battery_size

    def fuel_type(self):
        return "Electric Charge"    
    
        

my_tesla_car=ElectricCar("Tesla", "Model S", "100MW")

# print({my_tesla_car.get_brand(), my_tesla_car.battery_size, my_tesla_car.model})
# print(my_tesla_car.full_name())


print(isinstance(my_tesla_car,Car))
print(isinstance(my_tesla_car,ElectricCar))
my_car=Car("Toyota", "Corolla")
# my_car.model="holo"
# print(my_car.model)

my_car1=Car("TATA", "Nexon")
my_car2=Car("Marti", "Suzuki")

 



# print(my_car.brand)
# print(my_car.model)
# print(my_car.full_name())

# multiple inheritance
class Battery:
    def battery_info(self):
        return "this is battery"

class Engine:
    def engine_info(self):
        return "this is engine"

class ElectricCarTwo(Battery, Engine, Car):
    pass 

my_new_tesla=ElectricCarTwo("tesla", "Model X")

print(my_new_tesla.engine_info())
print(my_new_tesla.battery_info())




     