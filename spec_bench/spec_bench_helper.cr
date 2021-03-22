# NOTE: Some of the files in `spec_bench` folder use the 'spec' functionality.
# But, these files are more for testing the 'learning' capability, so some failures are assumed.
# For 'expect to succeed' files, see the regular 'spec' folder.

require "spec"
require "../src/ai4cr"
require "../spec/spec_helper"
require "../spec/test_helper"

QTY_X_PERCENT_DENOMINATOR = 20
# Be sure that MULTI_TYPE_TEST_QTY >= QTY_X_PERCENT_DENOMINATOR
# For more training, you'll probably want to dial up the "* 1" to "* 1000" or so.
MULTI_TYPE_TEST_QTY = QTY_X_PERCENT_DENOMINATOR * 5 # * 5 * 10

def histogram(arr, precision = 0) # , keys = [] of Float64)
  h = Hash(Float64, Int32).new
  arr.flatten.group_by { |v| v.round(precision) }
    .each { |elem| h[elem[0]] = elem[1].size }
  h.to_a.sort { |a, b| a[0] <=> b[0] }.to_h
end

def plot_errors(name, net)
  puts "\n--------\n"
  puts name

  min = 0.0
  max = 1.0
  precision = 2.to_i8
  in_bw = false
  prefixed = false
  reversed = false

  charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
  plot = charter.plot(net.error_stats.history, prefixed)

  puts "  plot: '#{plot}'"
  puts "  error_stats.history: '#{net.error_stats.history.map { |e| e.round(6) }}'"

  puts "\n--------\n"
end

def plot_weights(name, weights, verbose = false)
  puts "\n--------\n"
  puts name

  min = -1.0
  max = 1.0
  precision = 3.to_i8
  in_bw = false
  prefixed = false
  inverted_colors = true

  char_box = '\u2588' # 'x' # '\u25A0'

  bar_colors = [:red, :black, :dark_gray, :yellow, :light_gray, :white, :green]
  bar_chars = bar_colors.size.times.to_a.map { char_box }

  charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: inverted_colors)

  weights_flattened = weights.flatten
  puts "  TOTALS:: min: #{weights_flattened.min.round(precision*2)}, max: #{weights_flattened.max.round(precision*2)}, avg: #{(1.0 * weights_flattened.sum / weights_flattened.size).round(precision*2)}, stddev: #{weights_flattened.standard_deviation}"
  puts "  HISTOGRAM:: #{histogram(weights_flattened)}"
  puts "  ROWS::"
  weights.each do |row|
    plot = charter.plot(row, prefixed)

    puts "  plot: '#{plot}', min: #{row.min.round(precision*2)}, max: #{row.max.round(precision*2)}, avg: #{(1.0 * row.sum / weights_flattened.size).round(precision*2)}, stddev: #{row.standard_deviation}"
    puts "  row: '#{row.map { |e| e.round(precision*2) }}'" if verbose
  end

  puts "\n--------\n"
end

def mini_net_exp_best_guess(net, raw_in)
  net.eval(raw_in)
  net.guesses_best
end

def mini_net_relu_best_guess(net, raw_in)
  net.eval(raw_in)
  net.guesses_best
end
