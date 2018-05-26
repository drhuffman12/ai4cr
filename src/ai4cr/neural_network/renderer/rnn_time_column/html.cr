require "../../../prototype_rnn/rnn_time_column"
require "./generic"

# module PrototypeRnn
module Ai4cr
  module NeuralNetwork
    module Renderer
      module RnnTimeColumn
        class Html < Ai4cr::NeuralNetwork::Renderer::RnnTimeColumn::Generic
          def draw_wrapper_start
            # @txt = ""
            # @txt += "\n"
            # @txt += "=" * ASCII_HEADER_INDENT
            # net.structure.max.times { @txt += "-vvv-" }
            # @txt += "\n"
            @txt = "<table class='rtc_wrapper_start' border='1'>"
          end

          def draw_comments(n, ins, cur_layer_in_size, prev_hist, outs)
            @comments = "<caption style='text-align: left'>"
            super
            @comments += "</caption>"
          end

          def rnn_summary
            "<pre>" + super + "</pre>"
          end

          def draw_net_guessed
            # @txt += "net.guess_deltas_abs_avg: "
            # @txt +=  colored_value(val: net.guess_deltas_abs_avg, sigs: 4) + "\n"

            # @txt += "net.guess_error_abs: "
            # @txt +=  colored_value(val: net.guess_error_abs, sigs: 4) + "\n"

            @comments += "<ul>"
            @comments +=   "<li>"
            @comments +=     "net.name: #{net.name}"
            @comments +=   "</li>"
            @comments +=   "<li>"
            @comments +=     "net.structure: #{net.structure}"
            @comments +=   "</li>"
            @comments +=   "<li>"
            @comments +=     "net.guess_deltas_abs_avg: " + colored_value(tag: "span", val: net.guess_deltas_abs_avg, sigs: 4)
            @comments +=   "</li>"
            @comments +=   "<li>"
            @comments +=     "net.guess_error_abs: " + colored_value(tag: "span", val: net.guess_error_abs, sigs: 4)
            @comments +=   "</li>"
            @comments += "</ul>"
          end

          def draw_layer_debug(n, ins, cur_layer_in_size, prev_hist, outs)
            sigs = 3

            sub_txt =  "\nlayer: #{n}" + "\n"
            sub_txt +=  "\nlayer: #{n}, ins: #{ins.map {|i| "%+1.1f" % i}}" + "\n"
            sub_txt +=  "\nlayer: #{n}, cur_layer_in_size: #{cur_layer_in_size}" + "\n"
            sub_txt +=  "\nlayer: #{n}, prev_hist: #{prev_hist}" + "\n"
            # sub_txt +=  "\nlayer: #{n}, targets: #{net.targets.map {|o| "%+1.#{sigs}f" % o}}" + "\n"
            sub_txt +=  "\nlayer: #{n}, outs: #{outs.map {|o| "%+1.#{sigs}f" % o}}" + "\n"
            sub_txt += "\n"
            @comments += "<pre>" + sub_txt + "</pre>"
          end

          def draw_wrapper_end
            # @txt += "\n"
            # @txt += "=" * ASCII_HEADER_INDENT
            # net.structure.max.times { @txt += "-^^^-" }
            # @txt += "\n"
            @txt += "</table>"
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

          def draw_layer_output_labels(outs)
            # column indexes:
            sub_txt = "<th>" + colored_div("Outputs") + "</th>" # green # light_green

            outs.each_index do |j|
              sub_txt += "<td>#{j}</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          def draw_layer_header_targets
            sub_txt = ""
            # column output TARGETs:
            sub_txt = "<th>" + colored_div("TARGET", "black", "#77ff77") + "</th>" # green # light_green

            net.targets.each_with_index do |col_val, j|
              sub_txt += "<td>" + colored_value(tag: "div", val: col_val) + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          # def draw_layer_header_labels(weights_size_less_one, n, at_last_layer, outs, outs_size_less_one)
          #   # @txt += ("%+1.1f " % ins[i]) + ("%3s " % n) + ": "
          #   # sub_txt = ""

          #   # sub_txt += draw_layer_header_label(weights_size_less_one, n)
          #   sub_sub_txt = "GUESS(#{n < weights_size_less_one ? "hid" : "out"})"
          #   color_bkgd = n == weights_size_less_one ? "yellow" : "light_gray"
          #   sub_txt = "<th>" + colored_div(sub_sub_txt, "black", color_bkgd) + "</th>"

          #   outs.each_with_index do |col_val, j|
          #     out_txt = if at_last_layer || !(j == outs_size_less_one)
          #       colored_value(col_val)
          #     else
          #       formatted_float_value(col_val, "black", "#dddddd") # :light_gray)
          #     end
          #     sub_txt += "<td>" + out_txt + "</td>"
          #   end
          #   @txt += "<tr>" + sub_txt + "</tr>"
          # end

          def draw_layer_header_targets
            sub_txt = ""
            # column output TARGETs:
            sub_txt = "<th>" + colored_div("TARGET", "black", "#77ff77") + "</th>" # green # light_green

            net.targets.each_with_index do |col_val, j|
              sub_txt += "<td>" + colored_value(tag: "div", val: col_val) + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          def draw_layer_header_guesses(weights_size_less_one, n, at_last_layer, outs, outs_size_less_one)
            sub_sub_txt = "GUESS(#{n < weights_size_less_one ? "hid" : "out"})"
            color_bkgd = n == weights_size_less_one ? "yellow" : "light_gray"
            sub_txt = "<th>" + colored_div(sub_sub_txt, "black", color_bkgd) + "</th>"

            outs.each_with_index do |col_val, j|
              out_txt = if at_last_layer || !(j == outs_size_less_one)
                colored_value(tag: "div", val: col_val)
              else
                formatted_float_value("div", col_val, "black", "#dddddd") # :light_gray)
              end
              sub_txt += "<td>" + out_txt + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          # def draw_layer_header_label(weights_size_less_one, n)
          #   # column output ACTUALs:
          #   sub_sub_txt = "GUESS(#{n < weights_size_less_one ? "hid" : "out"})"
          #   color_fore = n == weights_size_less_one ? "yellow" : "light_gray"
          #   sub_txt = colored_div(sub_sub_txt, color_fore)
          #   "<th>" + sub_txt + "</th>"
          # end

          def draw_layer_best_guess(outs)
            # column output TARGETs:
            sub_txt = "<th>" + colored_div("BEST", "black", "orange") + "</th>" # light_yellow

            net.values_guessed_best.each_with_index do |col_val, j|
              sub_txt += "<td>" + colored_value(tag: "div", val: col_val) + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          def draw_layer_header_deltas_and_errors
            sub_txt = ""

            # column output DELTAs:
            sub_txt = "<th>" + colored_div("DELTA(out)", "black", "#aaaaff") + "</th>"

            net.output_deltas.each_with_index do |col_val, j|
              sub_txt += "<td>" + colored_value(tag: "div", val: col_val) + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"



            # column output ERRORs:
            sub_txt = "<th>" + colored_div("ERROR(out)", "black", "#ffaaaa") + "</th>"

            net.output_errors.each_with_index do |col_val, j|
              sub_txt += "<td>" + colored_value(tag: "div", val: col_val) + "</td>"
            end
            @txt += "<tr>" + sub_txt + "</tr>"
          end

          # def draw_inputs_and_weights(layer, ins, cur_layer_in_size, prev_hist)
          #   # Draw inputs and weights:
          #   layer_size_less_one = layer.size - 1
          #   layer.each_with_index do |layer_row, i|
          #     # @txt +=  "layer: #{layer}, i: #{i}"
          #     # @txt +=  "#{i}: #{layer_row}"
          #     # @txt +=  "i: #{i}"
          #     # val = i < ins.size ? ins[i] : prev_hist[i - cur_layer_in_size]
          #     # val = i < ins.size ? ins[i] : prev_hist[i - cur_layer_in_size]
          #     val = if i < ins.size
          #         ins[i]
          #       elsif i - cur_layer_in_size < prev_hist.size
          #         prev_hist[i - cur_layer_in_size]
          #       else
          #         0.0
          #       end

          #     if i == layer_size_less_one
          #       @txt += formatted_float_value(val, :light_gray)
          #     elsif i > layer_size_less_one
          #       @txt += ""
          #     else
          #       @txt += colored_value(val)
          #     end
          #     @txt += (" %3s " % i)
          #     @txt += ": "

          #     layer_row.each do |col_val|
          #       @txt += colored_value(col_val)
          #     end

          #     @txt += "\n"
          #   end
          # end

          # def draw_layer_separator
          #   @txt += "\n"
          #   @txt += "-" * ASCII_HEADER_INDENT
          #   net.structure.max.times { @txt += " --- " }
          #   @txt +=  "\n" + "\n"
          # end

          def colored_value(tag, val, sigs = 1)
            val_delta = 0.1
            color_bkgd = case
              when val < -1.0
                "#ffaaaa" # "red"
              when val > 1.0
                "#aaffaa" # "green"
              when nums_within_delta(val, -1.0, val_delta)
                "#ffdddd" # "light_red"
              when nums_within_delta(val, 1.0, val_delta)
                "#ddffdd" # "light_green"
              when nums_within_delta(val, 0.0, val_delta)
                "#aaaaff" # "blue"
              when val < 0.0
                "#ddddff" # "magenta"
              else
                "transparent" # "cyan"
            end
            formatted_float_value(tag, val, sigs, "black", color_bkgd)
          end

          # def formatted_float_value(val, color, sigs = 1)
          def formatted_float_value(tag, val, sigs = 1, color_fore = "black", color_bkgd = "transparent")
            # "<#{tag} style='color: #{color_fore}; background-color: #{color_bkgd}'>" + ("%+1.#{sigs}f" % val) + "<#{tag}>"
            val_str = "%+1.#{sigs}f" % val
            colored_tag(tag, val_str, color_fore, color_bkgd)
          end

          def colored_tag(tag, txt, color_fore = "black", color_bkgd = "transparent")
            "<#{tag} style='color: #{color_fore}; background-color: #{color_bkgd}'>#{txt}</#{tag}>"
          end

          def colored_div(txt, color_fore = "black", color_bkgd = "transparent")
            colored_tag("div", txt, color_fore, color_bkgd)
          end

          def colored_span(txt, color_fore = "black", color_bkgd = "transparent")
            colored_tag("span", txt, color_fore, color_bkgd)
          end
        end
      end
    end
  end
end
