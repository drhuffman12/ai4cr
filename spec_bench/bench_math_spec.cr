require "benchmark"
require "ascii_bar_charter"
require "./../src/ai4cr.cr"

height = 10 + rand(10) # .round
height_considering_bias = height + 1

Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("to_a") { (0..(height_considering_bias - 1)).to_a }
end

def equal_spaced_steps(min, max, step_max, precision)
  dist = max - min
  # step_max = bar_chars.size - 1
  (0..step_max).to_a.map { |i| (dist * i / step_max + min).round(precision) }
end

def equal_spaced_steps_shuffled(min, max, step_max, precision)
  equal_spaced_steps(min, max, step_max, precision).shuffle
end

def rnd(min, dist, precision)
  (min + rand(dist)).round(precision)
end

def random_steps(min, max, step_max, precision)
  dist = max - min
  # data = Array(Float64).new
  Array.new(step_max + 1) { rnd(min, dist, precision) }
  # Array(Float64).new(size: step_max).fill(rnd(min, dist, precision))
end

min = 0.0
max = 1.0
step_max = 10
precision = 3

# equal_spaced_steps(min, max, step_max, precision)
# # equal_spaced_steps(0.0, 1.0, 10, 3.to_i8)

# random_steps(min, max, step_max, precision)
# # random_steps(0.0, 1.0, 10, 3.to_i8)

# equal_spaced_steps_shuffled(min, max, step_max, precision)

[1, 10, 100, 1000, 10000].each do |sm|
  Benchmark.ips do |x|
    puts "step_max: #{sm}::"
    x.report("  equal_spaced_steps") { equal_spaced_steps(min, max, sm, precision) }
    x.report("  random_steps") { random_steps(min, max, sm, precision) }
    x.report("  equal_spaced_steps_shuffled") { equal_spaced_steps_shuffled(min, max, sm, precision) }
  end
end

# $ crystal spec spec_examples/ai4cr/neural_network/cmn/bench_math_spec.cr
# Warning: benchmarking without the `--release` flag won't yield useful results
# Array.new   4.44M (225.27ns) (± 1.97%)  128B/op        fastest
#      to_a   1.86M (538.18ns) (± 1.26%)  256B/op   2.39× slower
# Warning: benchmarking without the `--release` flag won't yield useful results
# step_max: 1::
#            equal_spaced_steps   4.44M (225.46ns) (± 1.61%)   112B/op        fastest
#                  random_steps   2.13M (469.58ns) (± 1.67%)  96.0B/op   2.08× slower
#   equal_spaced_steps_shuffled   2.47M (405.64ns) (± 2.23%)   176B/op   1.80× slower
# Warning: benchmarking without the `--release` flag won't yield useful results
# step_max: 10::
#            equal_spaced_steps   1.27M (790.43ns) (± 1.35%)  272B/op        fastest
#                  random_steps 475.49k (  2.10µs) (± 1.92%)  160B/op   2.66× slower
#   equal_spaced_steps_shuffled 558.01k (  1.79µs) (± 1.56%)  400B/op   2.27× slower
# Warning: benchmarking without the `--release` flag won't yield useful results
# step_max: 100::
#            equal_spaced_steps 171.21k (  5.84µs) (± 1.39%)  2.71kB/op        fastest
#                  random_steps  55.52k ( 18.01µs) (± 1.16%)  1.06kB/op   3.08× slower
#   equal_spaced_steps_shuffled  67.80k ( 14.75µs) (± 1.25%)  3.73kB/op   2.53× slower
# Warning: benchmarking without the `--release` flag won't yield useful results
# step_max: 1000::
#            equal_spaced_steps  18.95k ( 52.77µs) (± 1.10%)  20.6kB/op        fastest
#                  random_steps   5.73k (174.67µs) (± 2.46%)  7.89kB/op   3.31× slower
#   equal_spaced_steps_shuffled   7.29k (137.13µs) (± 1.58%)  28.4kB/op   2.60× slower
# Warning: benchmarking without the `--release` flag won't yield useful results
# step_max: 10000::
#            equal_spaced_steps   1.86k (537.99µs) (± 1.09%)   175kB/op        fastest
#                  random_steps 571.82  (  1.75ms) (± 1.27%)  78.2kB/op   3.25× slower
#   equal_spaced_steps_shuffled 732.85  (  1.36ms) (± 1.58%)   253kB/op   2.54× slower
