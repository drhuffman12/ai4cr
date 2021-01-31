module Ai4cr
  class Team(T)
    property team_size : Int32
    property team_last_id : Int32
    property training_round_qty : Int32
    property training_round_indexes : Array(Int32)
    getter team_indexes : Array(Int32)
    getter team_members : Array(T)
    # getter team_workers :

    property member_config

    def initialize(@team_size, @training_round_qty = 1, **member_config)
      @member_config = member_config
      @team_indexes = Array(Int32).new(team_size) { |i| i }
      @team_last_id = 0
      @team_members = @team_indexes.map do |i|
        named_config = @member_config.merge(name_instance: (team_last_id + i).to_s)
        T.new(**named_config)
      end
      @training_round_indexes = Array(Int32).new(training_round_indexes) { |i| i }
    end

    # def breed(delta = (rand*2 - 0.5)) : Array(T)
    def breed : Array(T)
      # breed in parallel
      channel = Channel(Int32).new
      new_team_members = Array(T).new

      # new_team_members =
      @team_members.each_with_index do |parent_a, ia|
        @team_members.each_with_index do |parent_b, ib|
          offspring = if ia == ib
                        parent_a
                      else
                        i = ia * @team_members.size + ib + 1
                        parent_a.class.breed(parent_a, parent_b, delta, name_instance: (team_last_id + i).to_s)
                      end
          # channel.send(1)
          channel.send(offspring)
        end
      end

      # sum = 0
      # @team_members.each do |member|
      #   sum += channel.receive
      # end

      sum = 0
      expecteed_sum = @team_members.size ** 2 # TODO: cache
      (1.expecteed_sum).each do |member|
        new_team_members << channel.receive
        sum += 1
      end

      raise "Missing a thread" if sum != expecteed_sum

      @team_last_id = @team_last_id + expecteed_sum

      new_team_members
    end

    def train(inputs_given, outputs_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT) # , breed_delta = (rand*2 - 0.5))
      # for X training rounds...
      @training_round_indexes.each do # |ri|
      # breed
        @team_members = breed # (breed_delta)

        # train in parallel
        channel = Channel(Int32).new

        @team_members.each do |member|
          spawn do
            member.train(inputs_given, outputs_expected, until_min_avg_error)
            channel.send(1)
          end
        end

        sum = 0
        @team_members.each do |member|
          sum += channel.receive
        end
        raise "Missing a thread" if sum != @team_members.Size

        # grade
        @team_members = @team_members.sort_by { |tm| tm.error_stats.distance }[0..team_size - 1]
      end
    end
  end
end
