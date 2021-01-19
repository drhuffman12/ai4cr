require "./../spec_helper"
require "./../spectator_helper"

Spectator.describe Ai4cr::Team do
  let(parents1) { (-10..0).to_a.map { |i| i/10.0 } }
  let(parents2) { (0..10).to_a.map { |i| i/10.0 } }

  let(breeder) {
    Ai4cr::Breeder.new
  }
end
