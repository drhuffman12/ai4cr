require "../../rnn_time_column_forward"

module Ai4cr
  module NeuralNetwork
    module Renderer
      module RnnTimeColumn
        abstract class Generic
          property rnn, eval_out_set, net, txt, file_path, comments

          def initialize(
            @rnn : Ai4cr::NeuralNetwork::Rnn,
            @eval_out_set : Array(Array(Float64)),
            @udc_in : Utils::DataConverter,
            @net : Ai4cr::NeuralNetwork::RnnTimeColumn,
            @file_path = "tmp/net_rnn_time_column.txt"
          )
            @txt = ""
            @comments = ""
            @dc = Utils::DataConverter.new(qty_steps: @net.output_size)
          end

          def draw(header_and_outputs_only = false)
            draw_wrapper_start
            weights_size_less_one = net.weights.size - 1
            if header_and_outputs_only
              draw_layer(weights_size_less_one, net.weights.last, net.weights.size - 1, header_and_outputs_only)
            else
              net.weights.each_with_index do |layer, n|
                draw_layer(weights_size_less_one, layer, n)
              end
            end
            # draw_net_guessed if net.guessed
            draw_wrapper_end
            log_and_save
            @txt
          end

          def log_and_save
            sigs = 3

            LOG.info "writing to file path '#{file_path}'"
            LOG.info "net.breeding_score: #{net.breeding_score}"
            dirname = File.dirname(file_path)
            Dir.mkdir_p(dirname) unless Dir.exists?(dirname)
            file_contents = @txt + "\n\n" + "net.breeding_score: #{net.breeding_score.map_with_index {|s, i| "%+1.#{(i + 1) * sigs}f" % s}}\n"
            # File.write(file_path, file_contents)
            File.open(file_path, "w+") { |f| f.print(file_contents) }
            # LOG.info file_contents
          end

          def draw_comments(n, ins, cur_layer_in_size, prev_hist, outs)
            draw_net_guessed if net.guessed
            draw_rnn_summary
            draw_layer_debug(n, ins, cur_layer_in_size, prev_hist, outs)
          end

          def rnn_summary
            sigs = 3

            sub_txt =  "\n"


            sub_txt += "  rnn.breeding_score   : #{rnn.breeding_score.map_with_index {|s, i| "%+1.#{(i + 1) * sigs}f" % s}}" + "\n" # first/top guess only
            sub_txt += "  eval_out_set         : #{eval_out_set}" + "\n"
            sub_txt += "  rnn.best_guesses     : #{rnn.best_guesses.map{|bg| bg.first}}" + "\n" # first/top guess only

            sub_txt += "  eval_out_set aka     : #{eval_out_set.map{|s_values| @udc_in.stepped_to_analog_value(s_values)}}" + "\n"
            sub_txt += "  rnn.best_guesses aka : #{rnn.best_guesses.map{|s_value_array| @udc_in.stepped_to_analog_value(s_value_array.first)}.map{|v| ("%+1.1f" % v).to_f64}}" + "\n" # first/top guess only

            sub_txt += "  matches              : #{rnn.best_guesses.map_with_index{|bg, i| bg.first == eval_out_set[i] ? '@' : '.'}}" + "\n" # first/top guess only
            sub_txt += "\n"
          end

          def draw_rnn_summary
            @comments += rnn_summary
          end

          def draw_layer(weights_size_less_one, layer, n, header_and_outputs_only = false)
            at_last_layer = n == weights_size_less_one
            ins = net.activation_nodes[n]
            cur_layer_in_size = net.activation_nodes[n].size
            prev_hist = (n+1 < net.activation_nodes_prev_t.size) ? net.activation_nodes_prev_t[n + 1] : [] of Float64
            outs = net.activation_nodes[n + 1]
            outs_size_less_one = outs.size - 1

            draw_comments(n, ins, cur_layer_in_size, prev_hist, outs)
            @txt += comments

            draw_layer_outputs(weights_size_less_one, n, at_last_layer, ins, cur_layer_in_size, prev_hist, outs, outs_size_less_one)


            draw_inputs_and_weights(layer, ins, cur_layer_in_size, prev_hist) if !header_and_outputs_only

            draw_layer_separator if net.structure.size > 2 && !at_last_layer
          end

          def draw_layer_outputs(weights_size_less_one, n, at_last_layer, ins, cur_layer_in_size, prev_hist, outs, outs_size_less_one)
            # draw_layer_debug(n, ins, cur_layer_in_size, prev_hist, outs)

            draw_layer_header_guesses(weights_size_less_one, n, at_last_layer, outs, outs_size_less_one)

            if at_last_layer && net.guessed
              draw_layer_best_guess(outs)
              draw_layer_header_targets
              draw_layer_header_deltas_and_errors
            end

            draw_layer_output_labels(outs)
          end

          # number formatting:

          def nums_within_delta(num_a, num_b, delta)
            (num_a - num_b).abs <= delta
          end

          ## to be implemented in sub-classes:

          def colored_value(val, sigs = 1)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_wrapper_start
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_net_guessed
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_wrapper_end
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_separator
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_debug(n, ins, cur_layer_in_size, prev_hist, outs)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_header_guesses(weights_size_less_one, n, at_last_layer, outs, outs_size_less_one)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_best_guess
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_header_targets
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_header_deltas_and_errors
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_layer_output_labels(outs)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          def draw_inputs_and_weights(layer, ins, cur_layer_in_size, prev_hist)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

          # number formatting:

          def formatted_float_value(val, color, sigs = 1)
            raise "Not Implemented: #{ {{@def.name.stringify}} }"
          end

        end
      end
    end
  end
end
