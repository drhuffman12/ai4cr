# require "../src/*"
# require "./spec_helper"
require "spectator"
require "./common_helper"
require "./test_helper_spectator"

Spectator.configure do |config|
  # config.fail_blank # Fail on no tests.
  # config.randomize  # Randomize test order.
  config.profile # Display slowest tests.
end
