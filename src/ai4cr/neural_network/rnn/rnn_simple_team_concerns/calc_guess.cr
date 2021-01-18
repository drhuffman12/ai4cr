module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleTeamConcerns
        module CalcGuess
          def eval(input_set_given)
            # Do an 'eval' on each of the team_members (in parallet?).
            channel = Channel(Int32).new

            team_members.each_with_index do |rnn_simple, i|
              spawn do # |rnn_simple, i|
                rnn_simple.eval(input_set_given)
                channel.send(i)
              end
            end

            sum = 0
            ids = Array(Int32).new

            team_size.times do
              id = channel.receive
              sum += id
              ids << id
            end

            # puts
            # puts "sum: #{sum}, ids: #{ids}"
            # puts

            team_members.map{|rnn_simple| rnn_simple.outputs_guessed}
          end

          def outputs_guessed
            team_members.map do |rnn_simple|
              rnn_simple.outputs_guessed
            end
          end
        end
      end
    end
  end
end
