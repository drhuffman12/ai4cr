abstract class Animal
end

class Dog < Animal
  def talk
    "Woof!"
  end
end

class Cat < Animal
  def talk
    "Miau"
  end
end

class Person
  getter pet

  def initialize(@name : String, @pet : Animal)
  end
end

class PetLover
  getter pets

  def initialize(@name : String, @pets : Array(Animal))
  end
end

john = Person.new "John", Dog.new
peter = Person.new "Peter", Cat.new
sally = PetLover.new "Sally", [Cat.new, Dog.new]

puts john.pet.talk
puts peter.pet.talk
puts sally.pets.map { |pet| pet.talk }

# Error: undefined method 'talk' for Person
