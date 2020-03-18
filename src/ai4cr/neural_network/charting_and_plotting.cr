require "ascii_bar_charter"

# include Ai4cr::NeuralNetwork::ChartingAndPlotting
module Ai4cr
  module NeuralNetwork
    module ChartingAndPlotting
      def histogram(arr, precision = 0)
        arr_flattened = arr.flatten
        hist = histogram_condensed(arr_flattened, precision)

        min = hist.keys.min
        max = hist.keys.max

        step_by = 10**precision
        full_keys = (min).step(by: step_by, to: max).to_a.map { |i| i.round(precision) }

        full_keys.each do |key|
          hist[key] ||= 0
        end

        hist
      end

      def histogram_condensed(arr, precision = 0) # , keys = [] of Float64)
        h = Hash(Float64, Int32).new
        # keys.each {|k| h[k] = 0}
        arr.flatten.group_by { |v| v.round(precision) }
          .each { |elem| h[elem[0]] = elem[1].size }
        h.to_a.sort { |a, b| a[0] <=> b[0] }.to_h
      end

      # def histogram_expanded(arr, precision = 0) # , keys = [] of Float64)
      #   h = Hash(Float64, Int32).new
      #   # keys.each {|k| h[k] = 0}
      #   arr.flatten.group_by { |v| v.round(precision) }
      #     .each { |elem| h[elem[0]] = elem[1].size }
      #   condensed = h.to_a.sort { |a, b| a[0] <=> b[0] }.to_h
      #   min = condensed.keys.min
      #   max = condensed.keys.max
      #   min.up_to(max).step(10 ** precision)

      # end

      def plot_histogram(name, values, precision)
        hist = histogram(values, precision)

        # puts "\n--------\n"
        puts name

        min = hist.values.min * 1.0
        max = hist.values.max * 1.0
        prec_i8 = precision.to_i8
        in_bw = false
        prefixed = false
        inverted_colors = false

        charter = AsciiBarCharter.new(min: min, max: max, precision: prec_i8, in_bw: in_bw, inverted_colors: inverted_colors)
        plot = charter.plot(hist.values, prefixed)

        puts "  Plot  :: '#{plot}'"
        puts "  X     :: min: '#{hist.keys.min}', max: '#{hist.keys.max}', step size: '#{precision}'"
        puts "  Y     :: min: '#{min}', max: '#{max}'"
        puts "  Counts:: #{hist}"

        # puts "\n--------\n"
      end

      def plot_errors(name, net)
        # puts "\n--------\n"
        puts name

        min = 0.0
        max = 1.0
        precision = 2.to_i8
        in_bw = false
        prefixed = false
        inverted_colors = false

        charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: inverted_colors)
        plot = charter.plot(net.error_distance_history, prefixed)

        puts "  plot: '#{plot}'"
        puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

        # puts "\n--------\n"
      end

      def plot_weights(name, weights, verbose = false)
        # puts "\n--------\n"
        puts name

        min = -1.0
        max = 1.0
        precision = 3.to_i8
        in_bw = false
        prefixed = false
        inverted_colors = true

        char_box = '\u2588' # 'x' # '\u25A0'
        # bar_chars = 11.times.to_a.map{ '\u25A0' }

        bar_colors = [:red, :black, :dark_gray, :yellow, :light_gray, :white, :green]
        # bar_chars = bar_colors.size.times.to_a.map{ '\u25A0' }
        bar_chars = bar_colors.size.times.to_a.map { char_box }

        charter = AsciiBarCharter.new(
          min: min, max: max, precision: precision,

          # bar_chars: AsciiBarCharter::BAR_CHARS,
          # bar_colors: AsciiBarCharter::BAR_COLORS,
          bar_chars: bar_chars,

          # bar_chars: AsciiBarCharter::BAR_CHARS_ALT,
          # bar_colors: AsciiBarCharter::BAR_COLORS_ALT,
          bar_colors: bar_colors,

          in_bw: in_bw, inverted_colors: inverted_colors
        )

        weights_flattened = weights.flatten
        puts "  TOTALS:: min: #{weights_flattened.min.round(precision*2)}, max: #{weights_flattened.max.round(precision*2)}, avg: #{(1.0 * weights_flattened.sum / weights_flattened.size).round(precision*2)}, stddev: #{weights_flattened.standard_deviation}"
        # puts "  HISTOGRAM:: #{histogram(weights_flattened)}"
        puts "  ROWS::"
        weights.each_with_index do |row, i|
          plot = charter.plot(row, prefixed)

          puts "  #{i.to_s.rjust(4, '0')}: '#{plot}', min: #{row.min.round(precision*2)}, max: #{row.max.round(precision*2)}, avg: #{(1.0 * row.sum / weights_flattened.size).round(precision*2)}, stddev: #{row.standard_deviation}"
          puts "  row: '#{row.map { |e| e.round(precision*2) }}'" if verbose
        end

        # puts "\n--------\n"
      end
    end
  end
end
