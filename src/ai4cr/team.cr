module Ai4cr
  class Team(T)
    property team_size : Int32
    getter team_indexes : Array(Int32)
    getter team_members : Array(T)

    property member_config

    def initialize(@team_size, member_config)
      @member_config = member_config
      @team_indexes = Array(Int32).new(team_size) { |i| i }
      @team_members = @team_indexes.map do |i|
        named_config = config.merge(name: i.to_s)
        t.new(**named_config)
      end
    end

    def breed
      @team_members.map_with_index do |parent_a, ia|
        @team_members.map_with_index do |parent_b, ib|
          return parent_a if ia == ib

          parent_a.class.breed(parent_a, parent_b)
        end
      end
    end
  end
end
