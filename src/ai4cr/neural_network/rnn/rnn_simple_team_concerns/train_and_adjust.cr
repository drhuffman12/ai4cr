module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleTeamConcerns
        module TrainAndAdjust
          def train(input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
            # Do an 'train' on each of the 'team_members' (in parallet?).
            breeded_team = breed
            # tm = team_members
            
            winning_team_members_scored = train_breeded_team(breeded_team, input_set_given, output_set_expected, until_min_avg_error)
            
            winning_indexes = winning_team_members_scored.map { |scored| scored[:index] }
            winning_members = winning_team_members_scored.map { |scored| scored[:member] }

            # TODO: track cache breeded_winners; for now, just puts them...
            puts
            # puts "winning_team_members_scored: #{winning_team_members_scored}"
            puts "winning_indexes: #{winning_indexes}"
            # puts "winning_members: #{winning_members}"
            # puts "team_members_scored.keys: #{team_members_scored.map{}}"
            puts

            @team_members = winning_members

            error_distance
          end

          def train_breeded_team(breeded_team, input_set_given, output_set_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
            channel = Channel(Int32).new

            breeded_team.each_with_index do |rnn_simple, i|
              spawn do # |rnn_simple, i|
                rnn_simple.train(input_set_given, output_set_expected, until_min_avg_error)
                channel.send(i)
              end
            end

            sum = 0
            ids = Array(Int32).new

            breeded_team.size.times do
              id = channel.receive
              sum += id
              ids << id
            end

            puts
            puts "sum: #{sum}, ids: #{ids}"
            puts

            calc_top_nets
          end

          def breed
            # Do an 'breed' on each of the 'team_members' (in parallet?) with each other.

            new_team_members = team_members # .map_with_index { |member, i| { i => member } }
            # new_team_members = # TODO

            new_team_members
          end

          def calc_top_nets
            # Sort each of the 'team_members' by @error_distance in ascending order.
            # Replace 'team_members' with the top 'team_size' qty winners.
            team_members.sort{ |a,b| a.error_distance <=> b.error_distance }[0..(team_size - 1)]

            team_members.map_with_index { |member, i| { index: i, member: member } }
              .sort{ |a,b| a[:member].error_distance <=> b[:member].error_distance }[0..(team_size - 1)]
          end
        end
      end
    end
  end
end
