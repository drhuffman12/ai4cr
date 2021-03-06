require "../../../../spec/spectator_helper"
require "../../../spec_bench_helper"
# require "../../../support/neural_network/data/*"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new }

  let(ancestor_adam_value) { 0.1 }
  let(ancestor_eve_value) { 0.9 }

  let(config_default_randomized) {
    Ai4cr::NeuralNetwork::Rnn::RnnSimple.new.config
  }

  let(config_adam) {
    config_default_randomized.merge(
      name: "Adam",

      bias_disabled: false,
      bias_default: (ancestor_adam_value / 2.0).round(1),

      learning_rate: ancestor_adam_value,
      # momentum: (1.0 - ancestor_adam_value).round(1),
      momentum: ancestor_adam_value,
      deriv_scale: (ancestor_adam_value / 4.0).round(1),

      history_size: 3
    )
  }

  let(config_eve) {
    config_default_randomized.merge(
      name: "Eve",

      bias_disabled: false,
      bias_default: (ancestor_eve_value / 2.0).round(1),

      learning_rate: ancestor_eve_value,
      # momentum: (1.0 - ancestor_eve_value).round(1),
      momentum: ancestor_eve_value,
      deriv_scale: (ancestor_eve_value / 4.0).round(1),

      history_size: 3
    )
  }
  let(ancestor_adam) {
    ancestor = my_breed_manager.create(**config_adam)
    ancestor.mini_net_set.each do |mini_net_li|
      mini_net_li.each do |mini_net_ti|
        mini_net_ti.weights.map_with_index! do |row, i|
          row.map_with_index! do |_col, j|
            (i + j / 10.0).round(1)
          end
        end
      end
    end
    ancestor.train(inputs, outputs)
    # ancestor.train(inputs, outputs)
    ancestor
  }
  let(ancestor_eve) {
    ancestor = my_breed_manager.create(**config_eve)
    ancestor.mini_net_set.each do |mini_net_li|
      mini_net_li.each do |mini_net_ti|
        mini_net_ti.weights.map_with_index! do |row, i|
          row.map_with_index! do |_col, j|
            -(i + j / 10.0).round(1)
          end
        end
      end
    end
    ancestor.train(inputs, outputs)
    # ancestor.train(inputs, outputs)
    ancestor
  }

  let(inputs) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }
  let(outputs) {
    [
      [0.1],
      [0.9],
    ]
  }

  let(inputs_sequence) {
    [
      [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
      [
        [0.3, 0.4],
        [0.5, 0.6],
      ],
      [
        [0.5, 0.6],
        [0.7, 0.8],
      ],
    ]
  }
  let(outputs_sequence) {
    [
      [
        [0.1],
        [0.9],
      ],
      [
        [0.9],
        [0.5],
      ],
      [
        [0.5],
        [0.3],
      ],
    ]
  }

  describe "#train_team" do
    it "successive generations score better (i.e.: lower errors)" do
      # TODO: (a) move to 'spec_bench' and (b) replace here with more 'always' tests
      max_members = 10
      qty_new_members = max_members

      params = Ai4cr::NeuralNetwork::Rnn::RnnSimple.new.config

      first_gen_members = my_breed_manager.build_team(qty_new_members, **params)
      second_gen_members = my_breed_manager.train_team(inputs, outputs, first_gen_members, max_members)
      third_gen_members = my_breed_manager.train_team(inputs, outputs, second_gen_members, max_members)

      first_gen_members_scored = first_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      first_gen_members_stats = first_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      second_gen_members_scored = second_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      second_gen_members_stats = second_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      third_gen_members_scored = third_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      third_gen_members_stats = third_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      puts
      puts "#train_team:"
      puts
      puts "first_gen_members_scored: #{first_gen_members_scored}"
      first_gen_members_stats.each { |m| puts m }

      puts
      puts "second_gen_members_scored: #{second_gen_members_scored}"
      second_gen_members_stats.each { |m| puts m }
      expect(second_gen_members_scored).to be < first_gen_members_scored

      puts
      puts "third_gen_members_scored: #{third_gen_members_scored}"
      third_gen_members_stats.each { |m| puts m }
      expect(third_gen_members_scored).to be < second_gen_members_scored
    end
  end

  describe "#train_team_using_sequence" do
    it "successive generations score better (i.e.: lower errors)" do
      # TODO: (a) move to 'spec_bench' and (b) replace here with more 'always' tests
      max_members = 10
      qty_new_members = max_members

      params = Ai4cr::NeuralNetwork::Rnn::RnnSimple.new.config

      first_gen_members = my_breed_manager.build_team(qty_new_members, **params)
      second_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, first_gen_members, max_members)
      third_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, second_gen_members, max_members)

      first_gen_members_scored = first_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      first_gen_members_stats = first_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      second_gen_members_scored = second_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      second_gen_members_stats = second_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      third_gen_members_scored = third_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
      third_gen_members_stats = third_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

      puts
      puts "#train_team_using_sequence:"
      puts
      puts "first_gen_members_scored: #{first_gen_members_scored}"
      first_gen_members_stats.each { |m| puts m }

      puts
      puts "second_gen_members_scored: #{second_gen_members_scored}"
      second_gen_members_stats.each { |m| puts m }
      expect(second_gen_members_scored).to be < first_gen_members_scored

      puts
      puts "third_gen_members_scored: #{third_gen_members_scored}"
      third_gen_members_stats.each { |m| puts m }
      expect(third_gen_members_scored).to be < second_gen_members_scored
    end
  end
end
