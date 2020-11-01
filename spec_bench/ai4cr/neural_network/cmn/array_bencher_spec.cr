require "benchmark"

# TODO: combine seeds re cross_net_bencher_spec.cr and array_bencher_spec.cr
height_considering_bias = 100 + 1

puts "\n==== compare Array.new vs to_a ====\n"
Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("to_a") { (0..(height_considering_bias - 1)).to_a }
end

puts "\n==== compare misc reversing of an array ====\n"
Benchmark.ips do |x|
  # x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("Array.new reversed") { Array.new(height_considering_bias) { |i| i }.reverse }
  x.report("Array.new max minus") { Array.new(height_considering_bias) { |i| height_considering_bias - i - 1 } }
  # x.report("to_a") { (0..(height_considering_bias - 1)).to_a.reverse }
  x.report("to_a reversed") { (0..(height_considering_bias - 1)).to_a.reverse }
  x.report("to_a downto") { (height_considering_bias - 1).downto(0).to_a.reverse }
end