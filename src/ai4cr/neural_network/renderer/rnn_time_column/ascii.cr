require "./generic"

module Ai4cr
  module NeuralNetwork
    module Renderer
      module RnnTimeColumn
        class Ascii < Ai4cr::NeuralNetwork::Renderer::RnnTimeColumn::Generic
          # TODO: convert to termbox (so can get 256 colors); see: https://github.com/andrewsuzuki/termbox-crystal/blob/master/examples/simple.cr

          ASCII_HEADER_INDENT = 12

          def draw_wrapper_start
            @txt = ""
            @txt += "\n"
            @txt += "=" * ASCII_HEADER_INDENT
            net.structure.max.times { @txt += "-vvv-" }
            @txt += "\n"
          end

          def draw_net_guessed
            @comments += "net.guess_deltas_abs_avg: "
            @comments +=  colored_value(val: net.guess_deltas_abs_avg, sigs: 4) + "\n"

            @comments += "net.guess_error_abs: "
            @comments +=  colored_value(val: net.guess_error_abs, sigs: 4) + "\n"
          end

          def draw_layer_debug(n, ins, cur_layer_in_size, prev_hist, outs)
            sigs = 6

            sub_txt =  "\nlayer: #{n}" + "\n"
            sub_txt +=  "\nlayer: #{n}, ins: #{ins.map {|i| "%+1.#{sigs}f" % i}}" + "\n"
            sub_txt +=  "\nlayer: #{n}, cur_layer_in_size: #{cur_layer_in_size}" + "\n"
            sub_txt +=  "\nlayer: #{n}, prev_hist: #{prev_hist}" + "\n"
            # sub_txt +=  "\nlayer: #{n}, targets: #{net.targets.map {|o| "%+1.#{sigs}f" % o}}" + "\n"
            sub_txt +=  "\nlayer: #{n}, outs: #{outs.map {|o| "%+1.#{sigs}f" % o}}" + "\n"
            sub_txt += "\n"
            @comments += sub_txt
          end

          def draw_wrapper_end
            @txt += "\n"
            @txt += "=" * ASCII_HEADER_INDENT
            net.structure.max.times { @txt += "-^^^-" }
            @txt += "\n"
          end

          # def draw(header_and_outputs_only = false)
          #   super
          #   log_and_save
          #   @txt
          # end

          # def log_and_save
          #   sigs = 3

          #   LOG.info "writing to file path '#{file_path}'"
          #   LOG.info "net.breeding_score: #{net.breeding_score}"
          #   dirname = File.dirname(file_path)
          #   Dir.mkdir_p(dirname) unless Dir.exists?(dirname)
          #   file_contents = @txt + "\n\n" + "net.breeding_score: #{net.breeding_score.map_with_index {|s, i| "%+1.#{(i + 1) * sigs}f" % s}}\n"
          #   # File.write(file_path, file_contents)
          #   File.open(file_path, "w+") { |f| f.print(file_contents) }
          #   # LOG.info file_contents
          # end

          # def draw_layer_outputs(weights_size_less_one, n, ins, cur_layer_in_size, prev_hist, outs)
          #   super
          # end

          def draw_layer_output_labels(outs)
            # column markers:
            @txt += " " * ASCII_HEADER_INDENT
            outs.each { |col_val| @txt += " ... " }
            @txt += "\n"

            # column indexes:
            @txt += " " * ASCII_HEADER_INDENT
            outs.each_index { |j| @txt += (" %3s " % j) }
            @txt += "\n"
          end

          def draw_layer_header_targets
            sub_txt = ""
            # column output TARGETs:
            sub_txt += ("%-#{ASCII_HEADER_INDENT - 2}s" % "TARGET").colorize(:black).on(:light_green).to_s
            sub_txt += ": "
            net.targets.each_with_index do |col_val, j|
              sub_txt += colored_value(col_val) + " "
            end
            sub_txt += "\n"
            @txt += sub_txt
          end

          def draw_layer_header_guesses(weights_size_less_one, n, at_last_layer, outs, outs_size_less_one)
            # @txt += ("%+1.1f " % ins[i]) + ("%3s " % n) + ": "

            # sub_txt += draw_layer_header_label(weights_size_less_one, n)
            sub_txt = "%-#{ASCII_HEADER_INDENT - 2}s" % "GUESS(#{n < weights_size_less_one ? "hid" : "out"})"
            sub_txt = case n
              when weights_size_less_one
                sub_txt = sub_txt.colorize(:yellow).to_s
              else
                sub_txt = sub_txt.colorize(:light_gray).to_s
              end
            sub_txt += ": "

            outs.each_with_index do |col_val, j|
              sub_txt += if at_last_layer || !(j == outs_size_less_one)
                colored_value(col_val) + " "
              else
                formatted_float_value(col_val, :light_gray) + " "
              end
            end
            sub_txt += "\n"
            @txt += sub_txt
          end

          # def draw_layer_header_label(weights_size_less_one, n)
          #   # column output ACTUALs:
          #   # @txt += " " * ASCII_HEADER_INDENT
          #   # @txt += ("%-#{ASCII_HEADER_INDENT - 2}s" % "GUESS(#{n < weights_size_less_one ? "hid" : "out"})").colorize(:yellow)
          #   sub_txt = "%-#{ASCII_HEADER_INDENT - 2}s" % "GUESS(#{n < weights_size_less_one ? "hid" : "out"})"
          #   sub_txt = case n
          #     when weights_size_less_one
          #       sub_txt = sub_txt.colorize(:yellow).to_s
          #     else
          #       sub_txt = sub_txt.colorize(:light_gray).to_s
          #     end
          #   sub_txt += ": "
          #   # @txt += sub_txt
          # end

          def draw_layer_best_guess(outs)
            # sub_txt = ""
            # sub_txt += @dc.top_n_stepped_values(outs, 1, to_ceil = true, to_floor = false).to_s
            # sub_txt += "\n" #XYZ\n"

            sub_txt = ""
            # column output TARGETs:
            sub_txt += ("%-#{ASCII_HEADER_INDENT - 2}s" % "BEST").colorize(:black).on(:light_yellow).to_s
            sub_txt += ": "

            # @dc.top_n_stepped_values(outs, 1, to_ceil = true, to_floor = true).each_with_index do |col_val, j|
            net.values_guessed_best.each_with_index do |col_val, j|
              sub_txt += colored_value(col_val) + " "
            end
            sub_txt += "\n"
            @txt += sub_txt
          end

          def draw_layer_header_deltas_and_errors
            sub_txt = ""

            # column output DELTAs:
            sub_txt += ("%-#{ASCII_HEADER_INDENT - 2}s" % "DELTA(out)").colorize(:blue).to_s
            sub_txt += ": "
            # sub_txt +=  "net.output_deltas: #{net.output_deltas}"
            net.output_deltas.each_with_index do |col_val, j|
            # net.guess_deltas_abs.each_with_index do |col_val, j|
              sub_txt += colored_value(col_val) + " "
            end
            sub_txt += "\n"

            # column output ERRORs:
            sub_txt += ("%-#{ASCII_HEADER_INDENT - 2}s" % "ERROR(out)").colorize(:red).to_s
            sub_txt += ": "
            # sub_txt +=  "net.output_errors: #{net.output_errors}"
            net.output_errors.each_with_index do |col_val, j|
              sub_txt += colored_value(col_val) + " "
            end
            sub_txt += "\n"
            @txt += sub_txt
          end

          def draw_inputs_and_weights(layer, ins, cur_layer_in_size, prev_hist)
            # Draw inputs and weights:
            layer_size_less_one = layer.size - 1
            layer.each_with_index do |layer_row, i|
              # @txt +=  "layer: #{layer}, i: #{i}"
              # @txt +=  "#{i}: #{layer_row}"
              # @txt +=  "i: #{i}"
              # val = i < ins.size ? ins[i] : prev_hist[i - cur_layer_in_size]
              # val = i < ins.size ? ins[i] : prev_hist[i - cur_layer_in_size]
              val = if i < ins.size
                  ins[i]
                elsif i - cur_layer_in_size < prev_hist.size
                  prev_hist[i - cur_layer_in_size]
                else
                  0.0
                end

              if i == layer_size_less_one
                @txt += formatted_float_value(val, :light_gray)
              elsif i > layer_size_less_one
                @txt += ""
              else
                @txt += colored_value(val) + " "
              end
              @txt += (" %3s " % i)
              @txt += ": "

              layer_row.each do |col_val|
                @txt += colored_value(col_val) + " "
              end

              @txt += "\n"
            end
          end

          def draw_layer_separator
            @txt += "\n"
            @txt += "-" * ASCII_HEADER_INDENT
            net.structure.max.times { @txt += " --- " }
            @txt +=  "\n" + "\n"
          end

          def colored_value(val, sigs = 1)
            val_delta = 0.1
            color = case
              when val < -1.0
                :red
              when val > 1.0
                :green
              when nums_within_delta(val, -1.0, val_delta)
                :light_red
              when nums_within_delta(val, 1.0, val_delta)
                :light_green
              when nums_within_delta(val, 0.0, val_delta)
                :blue
              when val < 0.0
                :magenta
              else
                :cyan
            end
            formatted_float_value(val, color, sigs)
          end

          def formatted_float_value(val, color, sigs = 1)
            ("%+1.#{sigs}f" % val).colorize(color || :black).to_s
          end
        end
      end
    end
  end
end
