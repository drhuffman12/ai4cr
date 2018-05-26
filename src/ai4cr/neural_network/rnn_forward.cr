require "./rnn_time_column_forward"

module Ai4cr
  module NeuralNetwork
    class RnnForward
      property name, rnn_time_col_size, rnn_time_cols, net_pos

      getter time_col_errors
      getter avg_errors

      getter state_and_top_sizes
      # getter to_ceil
      # getter to_floor
      getter time_col_guesses
      getter expected_outputs : Array(Array(Float64))
      getter disable_bias, learning_rate, momentum
      getter input_size, output_size, inner_sizes
      getter output_size, rnn_time_col_size, state_and_top_sizes, to_ceil, to_floor

      getter dc_in
      getter dc_out
      getter initial_weight_scale, top_guesses_extra_qty
      getter name_parent, name_prefix, name_suffix
      getter prev_rtc
      getter rounding_sigs

      # property breeding_history # : Array(NamedTuple(Symbol, Float64 | String | Array(Float64)))
      # @breeding_history : Array(NamedTuple(Symbol, Float64 | String | Array(Float64)))
      getter breeding_history_i_breed : Array(Int32)
      getter breeding_history_i_train : Array(Int32) # training_session
      getter breeding_history_net_pos : Array(String)
      getter breeding_history_score : Array(Array(Float64))

      property last_best_guesses_html : String
      # getter actual_outputs : Array(Array(Float64))

      # getter most_confident_guesses : Array(Array(Array(Int32 | Float64))) # <- most_confident_guesses # values_guessed_best_with_confidence
      # getter next_most_confident_guesses : Array(Array(Array(Int32 | Float64))) # <- next_most_confident_guesses # values_guessed_next_best_with_confidence
      # @force_hidden : Bool

      @name : String
      @prev_rtc : Ai4cr::NeuralNetwork::RnnTimeColumnForward?
      @rnn_time_cols : Array(Ai4cr::NeuralNetwork::RnnTimeColumnForward)

      @values_guessed_best_per_rtc : Array(Array(Float64))
      @values_guessed_correct_last_per_rtc : Float64
      @values_guessed_correct_all_per_rtc : Array(Float64)
      @values_guessed_correct_qty_per_rtc : Array(Float64)

      def clone(breeding_index, training_index, net_pos)
        self.class.new(
          disable_bias: self.disable_bias, learning_rate: self.learning_rate, momentum: self.momentum,
          input_size: self.input_size, output_size: self.output_size, inner_sizes: self.inner_sizes,
          rnn_time_col_size: self.rnn_time_col_size,
          state_and_top_sizes: self.state_and_top_sizes, to_ceil: self.to_ceil, to_floor: self.to_floor,

          dc_in: self.dc_in, dc_out: self.dc_out,

          initial_weight_scale: self.initial_weight_scale,
          top_guesses_extra_qty: self.top_guesses_extra_qty,
          name_parent: self.name_parent, name_prefix: self.name_prefix, name: self.name, name_suffix: self.name_suffix,

          breeding_index: breeding_index,
          training_index: training_index,
          net_pos: net_pos,
        )
      end

      def update_breeding_history(breeding_index, training_index, net_pos)
        @breeding_history_i_breed << breeding_index # .clone
        @breeding_history_i_train << training_index # .clone
        @breeding_history_net_pos << net_pos # .clone
        @breeding_history_score   << breeding_score # .clone # score.clone # breeding_score.clone
      end

      def initialize(
        @disable_bias : Bool = false, @learning_rate : Float64 = rand, @momentum : Float64 = rand,
        # @structure : Array(Int32),
        @input_size : Int32 = 2, @output_size : Int32 = 1, @inner_sizes : Array(Int32) = [0],
        @rnn_time_col_size : Int32 = 3,
        state_and_top_sizes : Array(Array(Int32))? = nil, @to_ceil = true, @to_floor = true,

        dc_in : Utils::DataConverter? = nil, dc_out : Utils::DataConverter? = nil,

        @initial_weight_scale : Float64 = 1.0, # 0.01
        @top_guesses_extra_qty : Int32 = 1,
        @name_parent = "", # TODO: remove!
        @name_prefix = "", # TODO: remove!
        name = "",
        @name_suffix = "",
        # @breeding_history = { -1 => {values_guessed_correct_qty: 0.0, guess_deltas_abs_avg: 0.0, guess_deltas_abs_last: 0.0, net_pos: "n/a"} }
        @breeding_index = -2,
        @training_index = -2,
        @net_pos = "n/a",
      )
        @last_best_guesses_html = ""
        @rounding_sigs = 6

        @breeding_history_i_breed = [breeding_index]
        @breeding_history_i_train = [training_index]
        @breeding_history_net_pos = [net_pos]
        @breeding_history_score   = [[0.0,0.0,0.0]]
        @expected_outputs = [[0.0]]
        # @actual_outputs = [[0.0]]

        # if name == ""
        #   @name = ""
        #   @name += name_prefix + "." if name_prefix.size > 0
        #   @name += ".col_shape_" + ([@input_size] + @inner_sizes + [@output_size].flatten.join("-")) + "."
        #   @name += "." + name_suffix if name_suffix.size > 0
        # else
        #   @name = name_parent # + name_prefix + name
        #   @name += ".col_shape_" + ([@input_size] + @inner_sizes + [@output_size].flatten.join("-")) + "."
        #   @name += name_suffix
        # end

        # @name = name_parent # + name_prefix + name
        @name = name # + name_prefix + name
        # @name += ".col_shape_" + (([@input_size] + @inner_sizes + [@output_size]).flatten.join("-")) + "."
        @name += name_suffix
        puts "CREATED RNN, name: #{@name}"

        @state_and_top_sizes = state_and_top_sizes.nil? ? [[output_size,1]] : state_and_top_sizes.as(Array(Array(Int32)))

        @time_col_errors = [0.0]
        @time_col_guesses = [[0.0]]
        @time_col_guesses_best = [[[0.0]]]
        @avg_errors = 0.0

        @values_guessed_best_per_rtc = [[0.0]]
        @values_guessed_correct_last_per_rtc = 0.0
        @values_guessed_correct_all_per_rtc = [0.0]
        @values_guessed_correct_qty_per_rtc = [0.0]

        @dc_in = dc_in.nil? ?  Utils::DataConverter.new(qty_steps: input_size) : dc_in.as(Utils::DataConverter)
        @dc_out = dc_out.nil? ?  Utils::DataConverter.new(qty_steps: output_size) : dc_out.as(Utils::DataConverter)

        @rnn_time_cols = [] of Ai4cr::NeuralNetwork::RnnTimeColumnForward
        @rnn_time_cols << Ai4cr::NeuralNetwork::RnnTimeColumnForward.new(
          disable_bias: @disable_bias, learning_rate: @learning_rate, momentum: @momentum,
          input_size: @input_size, output_size: @output_size, inner_sizes: @inner_sizes,
          prev_rtc: @prev_rtc,

          initial_weight_scale: @initial_weight_scale,
          top_guesses_extra_qty: @top_guesses_extra_qty,
          force_hidden: true,
          name_prefix: @name_prefix, # TODO: remove!
          name: @name,
          name_suffix: ".n01"
        )

        (1...@rnn_time_col_size).to_a.each do |i|
          @rnn_time_cols <<  Ai4cr::NeuralNetwork::RnnTimeColumnForward.new(
            disable_bias: @disable_bias, learning_rate: @learning_rate, momentum: @momentum,
            input_size: @input_size, output_size: @output_size, inner_sizes: @inner_sizes,
            prev_rtc: @prev_rtc,

            initial_weight_scale: @initial_weight_scale,
            top_guesses_extra_qty: @top_guesses_extra_qty,
            force_hidden: true,
            name_prefix: @name_prefix, # TODO: remove!
            name: @name,
            name_suffix: ".p#{"%02d" % i}"
          )
        end
      end

      def per_column_successes
        [@values_guessed_correct_last_per_rtc, @values_guessed_correct_all_per_rtc, @values_guessed_correct_qty_per_rtc, @rnn_time_col_size]
      end

      # def draw_best_ascii(eval_out_set, udc_in, header_and_outputs_only = false, file_path = nil)
      #   renderer_class = Ai4cr::NeuralNetwork::Renderer::RnnTimeColumn::Ascii
      #   @rnn_time_cols.map_with_index do |rtc, i_rtc|
      #     if rtc.values_guessed_correct_all == 1.0
      #       # timestamp = Time.now.to_utc.to_s.gsub(" ", ".").gsub(":", "-") + ".r_" + (10*rand).round(4).to_s
      #       winner_txt = i_rtc == @rnn_time_col_size - 1 ? "WINNER" : "good"
      #       file_path ||= "tmp/test.rnn_tri/#{@name}.#{winner_txt}.rtc_" + ("%02d" % i_rtc) + ".log"
      #       puts "#{winner_txt} file_path: #{file_path}"
      #       renderer = renderer_class.new(rnn: self, eval_out_set: eval_out_set, udc_in: udc_in, net: rtc, file_path: file_path)
      #       renderer.draw(header_and_outputs_only)
      #     end
      #   end
      # end

      # def draw_best_html(eval_out_set, udc_in, header_and_outputs_only = false, file_path = nil)
      #   renderer_class = Ai4cr::NeuralNetwork::Renderer::RnnTimeColumn::Html
      #   html_code = "<table><tr>"
      #   @rnn_time_cols.map_with_index do |rtc, i_rtc|
      #     if rtc.values_guessed_correct_all == 1.0
      #       # timestamp = Time.now.to_utc.to_s.gsub(" ", ".").gsub(":", "-") + ".r_" + (10*rand).round(4).to_s
      #       winner_txt = i_rtc == @rnn_time_col_size - 1 ? "WINNER" : "good"
      #       file_path ||= "tmp/test.rnn_tri/#{@name}.#{winner_txt}.rtc_" + ("%02d" % i_rtc) + ".log"
      #       puts "#{winner_txt} file_path: #{file_path}"
      #       renderer = renderer_class.new(rnn: self, eval_out_set: eval_out_set, udc_in: udc_in, net: rtc, file_path: file_path)
      #       renderer.draw(header_and_outputs_only)
      #       html_code += "<td>" + renderer.txt + "</td>"
      #     end
      #   end
      #   html_code += "</tr></table>"
      # end

      # def match_type(v_out, i_out, v_guess, v_best_guess, v_next_best_guess) # mcg_indexes, nmcg_indexes, mcg_confidences, nmcg_confidences)
      #   should_be = case
      #   when v_out == dc_out.max
      #     :should_be_on
      #   when v_out == dc_out.mid
      #     :is_unknown
      #   when v_out == dc_out.min
      #     :should_be_off
      #   else
      #     :between
      #   end

      #   confidence = guesses[i_out] # 0.0
      #   i_mcg_conf = mcg_indexes.index(i_out)
      #   i_nmcg_conf = nmcg_indexes.index(i_out)
    
      #   is_correct_guess = case
      #     when should_be == :should_be_on
      #       confidence = 0.0
      #       # i_mcg_conf.nil? ? 0.0 : mcg_confidences[i_mcg_conf.to_i32]
      #       # mcg_indexes.includes?(i_out)
      #       v_out == v_best_guess
      #     when should_be == :should_be_off
      #       confidence = 0.0
      #       # i_nmcg_conf.nil? ? 0.0 : nmcg_confidences[i_mcg_conf.to_i32]
      #       !nmcg_indexes.includes?(i_out)
      #       v_out == v_next_best_guess
      #     else
      #       false
      #     end

      #   is_next_guess = case
      #     when should_be == :should_be_on
      #       confidence = 0.0
      #       nmcg_indexes.includes?(i_out)
      #     when should_be == :should_be_off
      #       confidence = 0.0
      #       !nmcg_indexes.includes?(i_out)
      #     else
      #       false
      #     end

      #   {should_be: should_be, is_correct_guess: is_correct_guess, is_next_guess: is_next_guess, confidence: confidence}
      # end



      def confidence_to_color(should_be_on, confidence, is_correct_guess, is_close_guess)
        perc = dc_out.value_in_range_to_percent(confidence)
        color_should = should_be_on ? 0 : 127
        color_perc = (127 * perc).round.to_i32

        color_r = color_should # 255 # (255 * confidence).round.to_i32
        color_g = color_should # 255
        color_b = color_should # 255

        # color_a = 255

        color_g = 127 + color_perc if is_correct_guess
        # color_b = 127 + color_perc if is_correct_guess || is_close_guess
        # color_r += perc127 if !(is_correct_guess || is_close_guess)
        color_r = 127 + color_perc if !is_correct_guess
        # if is_correct_guess
        #   color_g += perc127 # 
        #   # color_g = 255 - (255 * perc).round.to_i32

        #   # color_g = (255 * perc).round.to_i32

        #   # color_r = 255 - (255 * perc).round.to_i32
        #   # color_b = 255 - (255 * perc).round.to_i32
        # elsif is_close_guess
        #   color_b += perc127 # 
        #   # color_b = 255 - (255 * perc).round.to_i32
          
        #   # color_b = (255 * perc).round.to_i32

        #   # color_r = 255 - (255 * perc).round.to_i32
        #   # color_g = 255 - (255 * perc).round.to_i32
        # else
        #   color_r += perc127 # 
        #   # color_r = 255 - (255 * perc).round.to_i32

        #   # color_r = (255 * perc).round.to_i32

        #   # color_g = 255 - (255 * perc).round.to_i32
        #   # color_b = 255 - (255 * perc).round.to_i32
        # end
        # color_r = 127 + (!is_correct_guess ? 127 * (perc) : 0).round.to_i32
        # # color_b = 255 - (is_close_guess ? 127 * (-perc) : 0).round.to_i32
        # # color_b = is_correct_guess ? 255 : (is_close_guess ? 170 : 85)
        # color_b = is_correct_guess ? 0 : (is_close_guess ? 63 : 127)
        # color_a = is_correct_guess ? 200 : (is_close_guess ? 100 : 50)
        # color_a = should_be_on && is_correct_guess ? 240 : (should_be_on && is_close_guess ? 160 : 80)
        color_a = should_be_on ? 1.0 : (!should_be_on ? 0.6 : 0.3)
        # color_a = should_be_on && is_correct_guess ? 80 : (should_be_on && is_close_guess ? 160 : 240)
        [color_r, color_g, color_b, color_a]
      end

      def update_last_best_guesses_html(eval_out_set : Array(Array(Float64))) # (html_tr)
        # raise "update_last_best_guesses_html, html_tr: '#{html_tr}'"
        # @last_best_guesses_html = html_tr.clone

        rtc = @rnn_time_cols.last # map_with_index do |rtc, i_rtc|
        i_rtc = @rnn_time_cols.size - 1
          eval_out = eval_out_set[i_rtc]
          mcg_indexes = rtc.most_confident_guesses.map { |g| g[0].round.to_i32 }
          nmcg_indexes = rtc.next_most_confident_guesses.map { |g| g[0].round.to_i32 }
          html_tr = confident_guesses_html_tr(eval_out, mcg_indexes, nmcg_indexes, rtc, i_rtc)
          # update_last_best_guesses_html(html_tr)
            @last_best_guesses_html = html_tr.clone # if i_rtc == @rnn_time_cols.size - 1
          #   # raise "YOU ARE HERE: draw_confident_guesses_html() .. wn_index: '#{wn_index}', i_rtc: '#{i_rtc}', last_best_guesses_html: '#{last_best_guesses_html}'"
          # end
          # html_code += html_tr
        # end
      end
      
      def draw_confident_guesses_html(folder, breeding_index, wn_index, net_pos, eval_out_set : Array(Array(Float64)), save_file = true, file_path = nil)
        # raise "draw_confident_guesses_html"
        html_code = "<table>" + confident_guesses_html_caption(breeding_index, wn_index, net_pos)

        html_code += confident_guesses_html_thead(".dt. \\ icol:")
        html_code += "<tbody>"

        # @last_best_guesses_html = html_code.clone
        
        # most_confident_guesses
        @rnn_time_cols.map_with_index do |rtc, i_rtc|
          eval_out = eval_out_set[i_rtc]
          mcg_indexes = rtc.most_confident_guesses.map { |g| g[0].round.to_i32 }
          nmcg_indexes = rtc.next_most_confident_guesses.map { |g| g[0].round.to_i32 }
          html_tr = confident_guesses_html_tr(eval_out, mcg_indexes, nmcg_indexes, rtc, i_rtc)
          # update_last_best_guesses_html(breeding_index, wn_index, html_tr) if i_rtc == rnn_time_col_size - 1
          # @last_best_guesses_html += "" + html_tr if i_rtc == rnn_time_col_size - 1
          # @last_best_guesses_html = html_tr # if i_rtc == @rnn_time_cols.size - 1
          # update_last_best_guesses_html(html_tr) if i_rtc == @rnn_time_cols.size - 1
          #   @last_best_guesses_html = html_tr.clone # if i_rtc == @rnn_time_cols.size - 1
          #   # raise "YOU ARE HERE: draw_confident_guesses_html() .. wn_index: '#{wn_index}', i_rtc: '#{i_rtc}', last_best_guesses_html: '#{last_best_guesses_html}'"
          # end
          html_code += html_tr
        end
        html_code += "</tbody>\n</table>"
        # @last_best_guesses_html += "</tbody>\n</table>"
        if save_file
          file_path ||= "#{folder}/#{@name}.breeding_index_" + ("%02d" % breeding_index) + ".html"
          puts "file_path: #{file_path}"
          File.write(file_path, html_code)

          # file_path ||= "#{folder}/#{@name}.breeding_index_" + ("%02d" % breeding_index) + ".last_col.html"
          puts "file_path: #{file_path + ".last_col.html"}"
          File.write(file_path + ".last_col.html", last_best_guesses_html)
        end
        html_code
      end
      
      def confident_guesses_html_caption(breeding_index, wn_index, net_pos)
        "<caption style='text-align: left'><pre>" +
          "* breeding_index: #{breeding_index}<br/>\n" +
          "* wn_index:       #{wn_index}<br/>\n" +
          "* net_pos:        #{net_pos}<br/>\n" +
          "* breeding_score: #{breeding_score}" +
          "</pre></caption>\n"
      end
      
      def confident_guesses_html_thead(left_most_label = ".. \\ icol:")
        html_code = "<thead>\n<tr><th>#{left_most_label}</th>"
        # (0...eval_out_set.first.size).to_a.each do |i_out|
        (0...output_size).to_a.each do |i_out|
          # eval_out = eval_out_set[i_out]
          # bkgd_color = confidence_to_color(1.0, is_correct_guess, is_next_guess)
          html_code += "<th>#{i_out}</th>"
        end
        html_code += "</tr>\n</thead>\n"
      end
      
      def confident_guesses_html_tr(eval_out, mcg_indexes, nmcg_indexes, rtc, i_rtc)
        html_code = "<tr><th title='breeding_score: #{breeding_score}'>#{i_rtc}</th>"

        eval_out.each_with_index do |v_out, i_out|
          lbl = "?"
          is_correct_guess = false
          is_close_guess = false
          is_a_best_guess = mcg_indexes.includes?(i_out)
          is_a_next_best_guess = nmcg_indexes.includes?(i_out)
          v_guess = rtc.values_guessed[i_out] #.map { |g| g[1].to_f64 }
          confidence = dc_out.value_in_range_to_percent(v_guess)

          should_be = case
          when v_out == dc_out.max
            # lbl = "x"
            is_correct_guess = is_a_best_guess
            is_close_guess = is_a_next_best_guess
            # lbl = is_correct_guess ? "X" : (is_close_guess ? "x" : "+")
            lbl = is_correct_guess ? "X" : (is_close_guess ? "^" : "+")
            :should_be_on
          when v_out == dc_out.mid
            lbl = "?"
            :is_unknown
          when v_out == dc_out.min
            # lbl = "o"
            is_correct_guess = !is_a_best_guess
            is_close_guess = !is_a_next_best_guess
            # lbl = is_correct_guess ? "O" : (is_close_guess ? "o" : ".")
            lbl = is_correct_guess ? "O" : (is_close_guess ? "v" : "-")
            :should_be_off
          else
            lbl = "%"
            :between
          end

          bkgd_color = confidence_to_color(should_be == :should_be_on, confidence, is_correct_guess, is_close_guess)
          # txt = should_be_on ? "X" : (is_unknown ? "?" : ".")  
          title = "* guess: #{v_out}, confidence: #{"%+1.#{rounding_sigs}f" % confidence},\n* is_a_best_guess: #{is_a_best_guess}, is_correct_guess: #{is_correct_guess},\n* is_a_next_best_guess: #{is_a_next_best_guess}, is_close_guess: #{is_close_guess}"
          style = "background-color: rgba(#{bkgd_color[0]}, #{bkgd_color[1]}, #{bkgd_color[2]}, #{bkgd_color[3]})"
          html_code += "  <td title='#{title}' style='#{style}'>#{lbl}</td>\n"
        end

        html_code += "</tr>\n"
      end
      
      # def update_last_best_guesses_html(breeding_index, wn_index, html_tr) # (folder, breeding_index, wn_index, net_pos, eval_out_set : Array(Array(Float64)), save_file = false, file_path = nil)
      #   @last_best_guesses_html = ""

      #   @last_best_guesses_html += ""
        
      # end

      def breeding_score
        # i.e.: roll-up: [@values_guessed_correct_qty, - @error] from each rtc
        [
          (@rnn_time_cols.map { |rtc| rtc.breeding_score[0] }.sum / @rnn_time_col_size),
          (@rnn_time_cols.map { |rtc| rtc.breeding_score[1] }.sum / @rnn_time_col_size),
          (@rnn_time_cols.map { |rtc| rtc.breeding_score[2] }.sum / @rnn_time_col_size)
        ]
      end

      def breeding_history
        breeding_history_i_breed.map_with_index do |i_breed, i|
          {
            i_breed: i_breed,
            i_train: breeding_history_i_train[i],
            net_pos: breeding_history_net_pos[i],
            score: breeding_history_score[i],
          }
        end
      end

      def train(inputs : Array(Array(Float64)), outputs : Array(Array(Float64))) # , to_ceil = true, to_floor = true
      # def train(io : Array(Float64), outputs : Array(Float64))
        # @time_col_guesses = [] of Array(Array(Float64))
        @time_col_errors = [] of Float64
        # @time_col_guesses_best = [] of Array(Array(Float64))
        @time_col_guesses = [] of Array(Float64)
        @time_col_guess_deltas = [] of Array(Float64)
        @time_col_guess_errors = [] of Array(Float64)

        @rnn_time_cols.each_with_index do |rtc, t|
          @time_col_errors << rtc.train(inputs[t], outputs[t])
          @time_col_guesses << rtc.activation_nodes.last.clone
          # @time_col_guesses_best << best_guesses # (to_ceil, to_floor)
        end

        @avg_errors = @time_col_errors.sum / @time_col_errors.size
        @time_col_errors
      end

      def best_guesses
        time_col_guesses.map do |time_col_guess|
          node_start_prev = 0
          state_and_top_sizes.map_with_index do |state_size_and_top_size, i|
            state_size = state_size_and_top_size[0]
            top_size  = state_size_and_top_size[1]
            node_start = node_start_prev
            node_end = node_start_prev + state_size - 1
            sub_guess = time_col_guess[node_start..node_end]
            sub_guess_best = dc_out.top_n_stepped_values(sub_guess, top_size, true, true) #, to_ceil, to_floor)
          end
        end
      end

      def eval(inputs : Array(Array(Float64)), expected_outputs : Array(Array(Float64)))
        @expected_outputs = expected_outputs
        # @actual_outputs = [] of Array(Float64)
        @time_col_guesses = [] of Array(Float64)
        @rnn_time_cols.each_with_index do |rtc, t|
          cur_col_guesses = rtc.eval(inputs[t], expected_outputs[t])
          # @time_col_guesses << cur_col_guesses
          @time_col_guesses << rtc.activation_nodes.last.clone

          @values_guessed_best_per_rtc << rtc.values_guessed_best
          # @values_guessed_correct_last_per_rtc << rtc.values_guessed_correct_last
          @values_guessed_correct_all_per_rtc << (rtc.values_guessed_correct_all ? 1.0 : 0.0)
          @values_guessed_correct_qty_per_rtc << 1.0 * rtc.values_guessed_correct_qty
        end
        @values_guessed_correct_last_per_rtc = @values_guessed_correct_all_per_rtc.last

        @time_col_guesses
      end
    end
  end
end
