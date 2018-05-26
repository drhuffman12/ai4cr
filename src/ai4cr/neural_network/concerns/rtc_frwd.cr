# Ai4cr::NeuralNetwork::Concerns::RnnFrwd
module Ai4cr
    module NeuralNetwork
      module Concerns
        module RtcFrwd
            ASCII_HEADER_INDENT = 12 # TODO remove render_ascii
            IO_MAX = 1.0
            IO_MID = 0.0
            IO_MIN = -1.0
        
            @name : String
            @activation_nodes_prev_t : Array(Array(Float64))
            @inputs : Array(Float64)
            @targets : Array(Float64)
            # @structure_from_prev_t : Array(Int32)
            # @structure_from_cur_t : Array(Int32)
        
            @initial_weight_scale : Float64
            @trained : Bool
            @guessed : Bool
        
            @top_guesses_extra_qty : Int32 # Float64
        
            # TODO: come up w/ more distinct names for these:
            @activation_node_deltas : Array(Array(Float64))
            @activation_node_errors : Array(Array(Float64))
            @guess_deltas_abs : Array(Float64)
            @guess_deltas_abs_avg : Float64
            @guess_error_abs : Float64
            @error : Float64
            @deltas_prev_t : Array(Array(Float64))
        
            @values_guessed : Array(Float64)
            @values_guessed_best : Array(Float64)
            @most_confident_guesses : Array(Array(Int32 | Float64))
            @next_most_confident_guesses : Array(Array(Int32 | Float64))
            # @values_guessed_correct_last : Bool
            @values_guessed_correct_all : Float64
            @values_guessed_correct_qty : Float64
            @values_guessed_correct_perc : Float64
        
            @poles_matches : Array(Bool)
        
            property inputs, targets, initial_weight_scale
            property activation_nodes_prev_t, input_size, output_size, inner_sizes, prev_rtc
        
            getter trained, guessed
            getter activation_node_deltas, activation_node_errors
            getter error
            getter output_deltas, output_errors
        
            getter values_guessed, poles_matches, values_guessed_best
            getter most_confident_guesses # values_guessed_best_with_confidence
            getter next_most_confident_guesses # values_guessed_next_best_with_confidence
            getter values_guessed_correct_all, values_guessed_correct_perc, values_guessed_correct_qty # , values_guessed_correct_last
            getter guess_deltas_abs, guess_error_abs
            getter guess_deltas_abs_avg
            getter name_prefix, name, name_suffix
        
            getter top_guesses_extra_qty, force_hidden, disable_bias
        
            def clone
              self.class.new(
                disable_bias: self.disable_bias, learning_rate: self.learning_rate, momentum: self.momentum,
                input_size: self.input_size, output_size: self.output_size, inner_sizes: self.inner_sizes,
                prev_rtc: self.prev_rtc,
        
                initial_weight_scale: self.initial_weight_scale,
                top_guesses_extra_qty: self.top_guesses_extra_qty,
                force_hidden: self.force_hidden,
                name_prefix: self.name_prefix, name: self.name, name_suffix: self.name_suffix
              )
            end
        
            def initialize(
                @disable_bias : Bool = false, @learning_rate : Float64 = rand, @momentum : Float64 = rand,
                # @structure : Array(Int32),
                @input_size : Int32 = 2, @output_size : Int32 = 1, @inner_sizes : Array(Int32) = [0],
                @prev_rtc : Ai4cr::NeuralNetwork::RnnTimeColumn? = nil,
                @initial_weight_scale : Float64 = 1.0, # 0.01
                @top_guesses_extra_qty : Int32 = 1, # to get the next x best guesses
                @force_hidden = true,
                @name_prefix = "", name = "", @name_suffix = ""
              )
              raise "Invalid Input Layer" if @input_size <= 0
              raise "Invalid Hidden Layers" if @force_hidden && @inner_sizes.map { |h| h <= 0 }.any?
              raise "Invalid Output Layer" if @output_size <= 0
        
              # if name == ""
              #   @name = ""
              #   @name += name_prefix + "." if name_prefix.size > 0
              #   @name += "col_shape_" +([input_size] + inner_sizes + [output_size]).flatten.join("-")
              #   @name += "." + name_suffix if name_suffix.size > 0
              # else
              #   @name = name_prefix + name + name_suffix
              # end
              @name = name_prefix + name + name_suffix
              puts "CREATED RTC, name: #{@name}"
        
              @activation_nodes_prev_t = [[IO_MID]]
        
              @structure = [@input_size]
              if @inner_sizes.map { |h| h <= 0 }.any?
                LOG.debug "SKIPPING hidden laters!"
              else
                LOG.debug "INCLUDING hidden laters!"
                @structure += @inner_sizes
              end
              @structure << @output_size
              @structure.compact!
        
              @inputs = (0...@output_size).to_a.map { IO_MID }
              @targets = (0...@output_size).to_a.map { IO_MID }
              @activation_node_deltas = (0...@output_size).to_a.map { [0.0] }
              @activation_node_errors = (0...@output_size).to_a.map { [0.0] }
        
              @deltas_prev_t = [[0.0]]
        
              @trained = false
              @guessed = false
        
              @top_guesses_target_qty = 0
              @values_guessed = [IO_MID]
              @values_guessed_best = [IO_MID]
              @most_confident_guesses = [[0, IO_MID]]
              @next_most_confident_guesses = [[0, IO_MID]]
              @poles_matches = [false]
              # @values_guessed_correct_last = false
              @values_guessed_correct_all = 0.0
              @values_guessed_correct_qty = 0.0
              @values_guessed_correct_perc = 0.0
        
              @output_deltas = (0...@structure.last).to_a.map { 0.0 }
              @output_errors = (0...@structure.last).to_a.map { 0.0 }
        
              @error = 0.0
        
              @guess_deltas_abs = [0.0]
              @guess_error_abs = 0.0
        
              @guess_deltas_abs_avg = 0.0
        
              @udc = Utils::DataConverter.new(qty_steps: output_size)
        
              super(structure: structure, disable_bias: @disable_bias, learning_rate: @learning_rate, momentum: @momentum)
            end
        
            def propagation_function
              # ->(x : Float64) { 1/(1 + ::Math.exp(-1*(x))) } # lambda { |x| ::Math.tanh(x) }
              ->(x : Float64) { ::Math.tanh(x) } # lambda { |x| ::Math.tanh(x) }
            end
        
            def derivative_propagation_function
              # ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
              ->(y : Float64) { 1.0 - y**2 } # lambda { |y| 1.0 - y**2 }
            end
        
            # Initialize neurons structure.
            def init_activation_nodes
              @activation_nodes = (0...@structure.size).to_a.map do |n|
                layer_size = @structure[n]
                nodes = (0...layer_size).to_a.map { n == 0 ? IO_MAX : IO_MID }
                nodes << IO_MAX if !(disable_bias || n == @structure.size - 1) # TODO: double-check condition
                # nodes << 10.0 * n + layer_size if !(disable_bias || n == @structure.size - 1)
                nodes
              end
              # if !(disable_bias || n == @structure.size - 1)
              #   @activation_nodes[0...-1].each { |layer| layer << 10.0 }
              # end
              @activation_nodes_prev_t = (0...@structure.size).to_a.map do |n|
                case
                  when n == 0 || n == @structure.size - 1 # input or output layer
                    [] of Float64
                  when @prev_rtc.is_a?(RnnTimeColumn) # hidden layer, prev_rtc is not nil
                    @prev_rtc.as(RnnTimeColumn).activation_nodes[n]
                  else # prev_rtc is nil
                    # IO_MID
                    # (0...@structure[n]).to_a.map { |i| -(10 * n + i) * 0.1 }
                    (0...@structure[n]).to_a.map { IO_MID }
                end
              end
              # @activation_nodes
        
              # puts
              # puts "activation_nodes_prev_t: #{activation_nodes_prev_t.pretty_inspect}"
              # puts
              # puts "activation_nodes: #{activation_nodes.pretty_inspect}"
              # puts
            end
        
            # Initialize the weight arrays using function specified with the
            # initial_weight_function parameter
            def init_weights
              @weights = (0...(@structure.size - 1)).to_a.map do |n|
                nodes_origin_size = @activation_nodes[n].size # 'from' might have bias
                if (n+1) < @activation_nodes_prev_t.size - 1
                  nodes_origin_size += @activation_nodes_prev_t[n+1].size
                end
                nodes_target_size = @structure[n + 1] # 'to' should not have bias
                (0...nodes_origin_size).to_a.map do |i|
                  (0...nodes_target_size).to_a.map do |j|
                    initial_weight_function.call(n, i, j) * initial_weight_scale
                  end
                end
              end
              # @weights.each do |w|
              #   w2 = w.map { |row| row.map { |val| val.round(1) } }
              # end
            end
        
            # Propagate values forward
            def feedforward(input_values)
              input_values.each_with_index do |elem, input_index|
                @activation_nodes.first[input_index] = elem # input_values[input_index]
              end
              if !@prev_rtc.nil?
                @activation_nodes_prev_t = @prev_rtc.as(Ai4cr::NeuralNetwork::RnnTimeColumn).activation_nodes.clone  # [1].each_with_index do |elem, input_index|
              end
        
              #   @activation_nodes_prev_t[1]
              #   @activation_nodes.first[@activation_nodes.first.size + input_index] = elem # @activation_nodes_prev_t[1][input_index]
              # end
              @weights.each_with_index do |_elem, n|
                @structure[n + 1].times do |j|
                  sum = 0.0
                  @activation_nodes[n].each_with_index do |_elem, i|
                    sum += (@activation_nodes[n][i] * @weights[n][i][j])
                  end
                  cur_layer_in_size = @activation_nodes[n].size
                  if n + 1 < activation_nodes_prev_t.size - 1
                    @activation_nodes_prev_t[n+1].each_with_index do |_elem, i|
                      if cur_layer_in_size + i < @weights[n].size
                        LOG.debug ""
                        LOG.debug "structure: #{structure}, n: #{n}, n+1: #{n+1}, i: #{i}, j: #{j}, cur_layer_in_size: #{cur_layer_in_size}"
                        LOG.debug "n+1: #{n+1}, cur_layer_in_size + i: #{cur_layer_in_size + i}"
                        LOG.debug "@weights.size: #{@weights.size}"
                        LOG.debug "@weights[n].size: #{@weights[n].size}"
                        LOG.debug "@weights[n][cur_layer_in_size + i].size: #{@weights[n][cur_layer_in_size + i].size}"
                        LOG.debug "@weights[n][cur_layer_in_size + i][j]: #{@weights[n][cur_layer_in_size + i][j]}"
                        LOG.debug ""
                        LOG.debug "@activation_nodes_prev_t.size: #{@activation_nodes_prev_t.size}"
                        LOG.debug "@activation_nodes_prev_t[n+1].size: #{@activation_nodes_prev_t[n+1].size}"
                        LOG.debug ""
        
                        sum += (@activation_nodes_prev_t[n + 1][i] * @weights[n][cur_layer_in_size + i][j])
                      end
                    end
                  end
                  @activation_nodes[n + 1][j] = propagation_function.call(sum)
                end
              end
            end
        
            # Calculate deltas for output layer
            def calculate_activation_node_deltas(expected_values)
              output_values = @activation_nodes.last
              # puts "@activation_nodes: #{@activation_nodes}"
              # puts "expected_values: #{expected_values}"
              # puts "output_values: #{output_values}"
        
              activation_node_deltas = [] of Float64
              output_values.each_with_index do |_elem, output_index|
                error = expected_values[output_index] - output_values[output_index]
                activation_node_deltas << derivative_propagation_function.call(output_values[output_index]) * error
              end
              @deltas = [activation_node_deltas]
            end
        
        
            # Calculate deltas for hidden layers
            def calculate_internal_deltas
              @activation_node_errors = [] of Array(Float64)
              @activation_node_deltas = [] of Array(Float64)
              prev_deltas = @deltas.last
              (@activation_nodes.size - 2).downto(1) do |layer_index|
                cur_layer_in_size = @activation_nodes[layer_index].size
                layer_deltas = [] of Float64
                node_errors = [] of Float64
        
                @activation_nodes[layer_index].each_with_index do |_elem, i|
                  error = 0.0
                  @structure[layer_index + 1].times do |j|
                    error += prev_deltas[j] * @weights[layer_index][i][j]
                  end
                  node_errors << error
                  layer_deltas << (derivative_propagation_function.call(@activation_nodes[layer_index][i]) * error)
                end
        
                # TODO: (?) backprop to prev time columns
                prev_layer_index = layer_index + 1
                if prev_layer_index < @activation_nodes_prev_t.size
                  @activation_nodes_prev_t[prev_layer_index].each_with_index do |_elem, i|
                    i2 = i + @activation_nodes[layer_index].size
                    error = 0.0
                    if prev_layer_index < @weights.size
                      @structure[layer_index + 1].times do |j|
        
                        LOG.debug ""
                        LOG.debug "prev_layer_index: #{prev_layer_index}, i2: #{i2}, j: #{j}"
                        LOG.debug "@weights.size: #{@weights.size}"
                        LOG.debug "@weights[layer_index].size: #{@weights[layer_index].size}"
                        LOG.debug "@weights[layer_index][i2].size: #{@weights[layer_index][i2].size}"
                        LOG.debug "@weights[layer_index][i2][j]: #{@weights[layer_index][i2][j]}"
                        LOG.debug ""
        
                        w = @weights[layer_index][i2][j]
                        d = prev_deltas[j]
                        error += d * w
                      end
                    end
                    node_errors << error
                    layer_deltas << (derivative_propagation_function.call(@activation_nodes_prev_t[prev_layer_index][i]) * error)
                  end
                end
        
                @activation_node_errors << node_errors
                @activation_node_deltas << layer_deltas
                prev_deltas = layer_deltas
                @deltas.unshift(layer_deltas)
              end
              @activation_node_errors.reverse!
              @activation_node_deltas.reverse!
            end
        
            # Update weights after @deltas have been calculated.
            # NOTE:
            def update_weights
              (@weights.size - 1).downto(0) do |n|
                @weights[n].each_with_index do |_elem, i|
                  cur_layer_in_size = @activation_nodes[n].size
                  @weights[n][i].each_with_index do |_elem, j|
                    # change = @deltas[n][j]*@activation_nodes[n][i]
                    # node_value = i < cur_layer_in_size ? @activation_nodes[n][i] : @activation_nodes_prev_t[n + 1][i - cur_layer_in_size]
        
                    node_value = if i < cur_layer_in_size
                        @activation_nodes[n][i]
                      elsif n < @activation_nodes_prev_t.size - 2 && (i - cur_layer_in_size) < @activation_nodes_prev_t[n + 1].size
                        @activation_nodes_prev_t[n + 1][i - cur_layer_in_size]
                      else
                        0.0 # ?: 0 or IO_MID
                      end
        
                    change = @deltas[n][j]*node_value
        
                    @weights[n][i][j] += (learning_rate * change +
                                          momentum * @last_changes[n][i][j])
                    @last_changes[n][i][j] = change
                  end
                end
              end
            end
        
            # Calculate quadratic error for a expected output value
            # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
            def calculate_error(expected_output)
              @output_deltas = [] of Float64
              @output_errors = [] of Float64
              @guess_deltas_abs = [] of Float64
              output_values = @activation_nodes.last
              error = 0.0
              expected_output.each_with_index do |_elem, output_index|
                delta = 1.0 * output_values[output_index] - expected_output[output_index]
                err = 0.5*(delta)**2
                @output_deltas << delta
                @guess_deltas_abs << delta.abs
                @output_errors << err
                error += err
              end
              @guess_deltas_abs_avg = @guess_deltas_abs.sum / guess_deltas_abs.size
              @error = error
              return error
            end
        
            # This method trains the network using the backpropagation algorithm.
            #
            # input: Networks input
            #
            # output: Expected output for the given input.
            #
            # This method returns the network error:
            # => 0.5 * sum( (expected_value[i] - output_value[i])**2 )
            def train(inputs, outputs)
              @inputs = inputs.map { |v| v.to_f }
              @targets = outputs.map { |v| v.to_f }
              outputs = @targets.clone
              eval(@inputs, @targets)
              backpropagate(outputs) # TODO: un-comment
              err = calculate_error(outputs)
              @trained = true
              err
            end
        
            getter top_guesses_target_qty
            
            # Evaluates the input.
            # E.g.
            #     net = Backpropagation.new([4, 3, 2])
            #     net.eval([25, 32.3, 12.8, 1.5])
            #         # =>  [0.83, 0.03]
            def eval(inputs, targets = @structure.last.times.to_a.map { IO_MID })
              @inputs = inputs.map { |v| v.to_f }
              @targets = targets.map { |v| v.to_f }
              check_input_dimension(@inputs.size)
              init_network if !@weights
              feedforward(@inputs)
              @guessed = true
        
              @top_guesses_target_qty = @targets.map { |target| target == IO_MAX ? 1 : 0 }.sum
              @values_guessed = @activation_nodes.last.clone
              # @values_guessed_best = @udc.top_n_stepped_values(@values_guessed, @top_guesses_extra_qty, to_ceil = true, to_floor = true)
        
              @values_guessed_best = @udc.top_n_stepped_values(@values_guessed, top_guesses_target_qty, to_ceil: true, to_floor: true)
              # @values_guessed_correct_last = @targets.last == @values_guessed_best.last
              top_guesses_all_qty = top_guesses_target_qty + @top_guesses_extra_qty
              most_confident_guesses = @udc.top_n_stepped_values_with_confidence(@values_guessed, top_guesses_all_qty, to_ceil: true, to_floor: true)
              @most_confident_guesses = most_confident_guesses[0...top_guesses_target_qty]
              @next_most_confident_guesses = most_confident_guesses[top_guesses_target_qty...top_guesses_all_qty]
        
              @poles_matches = @udc.to_pole_and_compare_signs(targets, @values_guessed)
        
              # # TODO: Fix? (This will move any any incorrect guesses .. moving them from an incorrect '+1' to a correct '-1', but then there are fewer '+1' guesses than expected)
              # # TODO: Maybe bump the guesses to IO_MAX and ignore the poles?
              # @compare_signs = false # true
              # if @compare_signs
              #   @poles_matches.each_with_index do |pm, i_pm|
              #     @values_guessed_best[i_pm] = IO_MIN unless pm
              #   end
              # end
        
              @values_guessed_correct_all = @targets == @values_guessed_best ? 1.0 : 0.0
              @values_guessed_correct_qty = @targets.map_with_index { |target, j| target == IO_MAX && target == @values_guessed_best[j] ? 1.0 : 0.0 }.sum
              # @values_guessed_incorrect_qty = @targets.map_with_index { |target, j| target != IO_MAX && target == @values_guessed_best[j] ? 1.0 : 0.0 }.sum
        
              @values_guessed_correct_perc = @top_guesses_target_qty == 0 ? 0.0 : 100.0 * @values_guessed_correct_qty / @top_guesses_target_qty
        
              @error = calculate_error(targets)
        
              return @values_guessed
            end
        
            def breeding_score
              # [@values_guessed_correct_qty, - @error]
              [values_guessed_correct_qty, - guess_deltas_abs_avg, - error / structure.last]
            end
        
            # def scores
            #   {
            #     values_guessed_correct_all: @values_guessed_correct_all,
            #     values_guessed_correct_perc: @values_guessed_correct_perc,
            #     values_guessed_correct_qty: @values_guessed_correct_qty,
            #     top_guesses_extra_qty: @top_guesses_target_qty, # @top_guesses_extra_qty,
            #     output_size: @output_size,
            #     error: @error
            #   }
            # end
        
            # Custom serialization. It used to fail trying to serialize because
            # it uses lambda functions internally, and they cannot be serialized.
            # Now it does not fail, but if you customize the values of
            # * initial_weight_function
            # * propagation_function
            # * derivative_propagation_function
            # you must restore their values manually after loading the instance.
            def marshal_dump
        
            end
        
            def marshal_load(tup)
            end        
        end
    end
end
