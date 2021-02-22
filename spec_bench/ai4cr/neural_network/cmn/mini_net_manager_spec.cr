# Yeah, technically not a spec, but let's roll with this for now ...
require "benchmark"
require "ascii_bar_charter"
require "../../../../spec/spectator_helper"
require "../../../spec_bench_helper"
require "../../../support/neural_network/data/*"

def train_and_and_cross_breed_team_set_using_sequence(
  my_breed_manager, first_gen_members,
  inputs_sequence_train, outputs_sequence_train,
  and_cross_breed = true
)
  is_error_decreasing = Array(Bool).new

  max_members = 10
  qty_new_members = max_members

  puts_debug
  puts_debug "v"*80
  puts_debug "#train_team_using_sequence with learning_style: (#{first_gen_members.first.learning_style}), and_cross_breed: #{and_cross_breed}"

  # breed and train
  # first_gen_members = my_breed_manager.build_team(qty_new_members, **params)
  first_gen_members_clones = first_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

  second_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence_train, outputs_sequence_train, first_gen_members_clones, max_members, and_cross_breed: and_cross_breed)
  second_gen_members_clones = second_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

  third_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence_train, outputs_sequence_train, second_gen_members_clones, max_members, and_cross_breed: and_cross_breed)
  third_gen_members_clones = third_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

  fourth_gen_members_clones = my_breed_manager.train_team_using_sequence(inputs_sequence_train, outputs_sequence_train, third_gen_members_clones, max_members, and_cross_breed: and_cross_breed)

  # training scores
  first_gen_members_train_scored = first_gen_members_clones.map { |member| member.error_stats.score }.sum / qty_new_members
  first_gen_members_train_stats = first_gen_members_clones.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

  second_gen_members_train_scored = second_gen_members_clones.map { |member| member.error_stats.score }.sum / qty_new_members
  second_gen_members_train_stats = second_gen_members_clones.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

  third_gen_members_train_scored = third_gen_members_clones.map { |member| member.error_stats.score }.sum / qty_new_members
  third_gen_members_train_stats = third_gen_members_clones.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

  fourth_gen_members_train_scored = fourth_gen_members_clones.map { |member| member.error_stats.score }.sum / qty_new_members
  fourth_gen_members_train_stats = fourth_gen_members_clones.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

  # training scores output
  puts_debug
  puts_debug "first_gen_members_train_scored: #{first_gen_members_train_scored}"
  first_gen_members_train_stats.each { |m| puts_debug m }

  puts_debug
  puts_debug "second_gen_members_train_scored: #{second_gen_members_train_scored}"
  second_gen_members_train_stats.each { |m| puts_debug m }
  # expect(second_gen_members_train_scored).to be < first_gen_members_train_scored
  is_error_decreasing << (second_gen_members_train_scored < first_gen_members_train_scored)

  puts_debug
  puts_debug "third_gen_members_train_scored: #{third_gen_members_train_scored}"
  third_gen_members_train_stats.each { |m| puts_debug m }
  # expect(third_gen_members_train_scored).to be < second_gen_members_train_scored
  is_error_decreasing << (second_gen_members_train_scored < first_gen_members_train_scored)

  puts_debug
  puts_debug "fourth_gen_members_train_scored: #{fourth_gen_members_train_scored}"
  fourth_gen_members_train_stats.each { |m| puts_debug m }
  # expect(fourth_gen_members_train_scored).to be < second_gen_members_train_scored
  is_error_decreasing << (second_gen_members_train_scored < first_gen_members_train_scored)

  # puts_debug "is_error_decreasing: #{is_error_decreasing}"
  # puts_debug "is_error_decreasing.all?: #{is_error_decreasing.all?}"

  puts_debug "-"*80

  {
    is_error_decreasing:    is_error_decreasing,
    members_per_generation: [first_gen_members_clones, second_gen_members_clones, third_gen_members_clones, fourth_gen_members_clones],
  }
end

def eval_team_set_using_sequence(my_manager, team_members_set_generation, inputs_sequence_for_guessing, outputs_sequence_expected)
  puts_debug "  learning_style: #{team_members_set_generation.first.first.learning_style}"

  inputs_sequence_for_guessing.each_with_index do |inputs, i|
    outputs_expected = outputs_sequence_expected[i]
    puts_debug "    outputs_expected: #{outputs_expected}"

    team_gen_scores = team_members_set_generation.map_with_index do |team_members, j|
      error_summaries = team_members.map_with_index do |member, k|
        guess = member.eval(inputs)

        output_errors = guess.map_with_index do |og, l|
          outputs_expected[l] - og
        end

        error_distance = output_errors.map { |e| 0.5 * e ** 2 }.sum

        score = [error_distance].map_with_index do |e, i|
          e / (2.0 ** ([error_distance].size - i))
        end.sum.round(4)

        score
      end

      # guesses = my_manager.eval_team_in_parallel(team_members, inputs)

      # error_summaries = guesses.map do |guess|
      #   output_errors = guess.map_with_index do |og, l|
      #     outputs_expected[l] - og
      #   end

      #   error_distance = output_errors.map { |e| 0.5 * e ** 2 }.sum

      #   score = [error_distance].map_with_index do |e, i|
      #     e / (2.0 ** ([error_distance].size - i))
      #   end.sum.round(4)

      #   score
      # end

      team_score = error_summaries.map_with_index do |e, i|
        e / (2.0 ** (error_summaries.size - i))
      end.sum.round(4)
      puts_debug "      team_score: #{team_score}"

      error_summaries
    end
  end
end

Spectator.describe Ai4cr::NeuralNetwork::Cmn::MiniNetManager do
  context "using triangle-square-cross example data" do
    let(inputs_triangle) { TRIANGLE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_square) { SQUARE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_cross) { CROSS.flatten.map { |input| input.to_f / 10.0 } }

    let(inputs_triangle_with_noise) { TRIANGLE_WITH_NOISE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_square_with_noise) { SQUARE_WITH_NOISE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_cross_with_noise) { CROSS_WITH_NOISE.flatten.map { |input| input.to_f / 10.0 } }

    let(inputs_triangle_base_noise) { TRIANGLE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_square_base_noise) { SQUARE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 10.0 } }
    let(inputs_cross_base_noise) { CROSS_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 10.0 } }

    let(expected_outputs_triangle) { [1.0, 0.0, 0.0] }
    let(expected_outputs_square) { [0.0, 1.0, 0.0] }
    let(expected_outputs_cross) { [0.0, 0.0, 1.0] }

    let(inputs_size) { inputs_triangle.size }
    let(outputs_size) { expected_outputs_triangle.size }

    let(io_sequence_train) {
      [
        [inputs_triangle, expected_outputs_triangle],
        [inputs_square, expected_outputs_square],
        [inputs_cross, expected_outputs_cross],
      ].shuffle
    }
    let(inputs_sequence_train) { io_sequence_train.map { |io| io[0] } }
    let(outputs_sequence_train) { io_sequence_train.map { |io| io[1] } }

    let(io_sequence_guess) {
      [
        [inputs_triangle, expected_outputs_triangle], [inputs_triangle_with_noise, expected_outputs_triangle], [inputs_triangle_base_noise, expected_outputs_triangle],
        [inputs_square, expected_outputs_square], [inputs_square_with_noise, expected_outputs_square], [inputs_square_base_noise, expected_outputs_square],
        [inputs_cross, expected_outputs_cross], [inputs_cross_with_noise, expected_outputs_cross], [inputs_cross_base_noise, expected_outputs_cross],
      ]
    }
    let(inputs_sequence_for_guessing) {
      [
        inputs_triangle, inputs_triangle_with_noise, inputs_triangle_base_noise,
        inputs_square, inputs_square_with_noise, inputs_square_base_noise,
        inputs_cross, inputs_cross_with_noise, inputs_cross_base_noise,
      ]
    }
    let(outputs_sequence_expected) { io_sequence_guess.map { |io| io[1] } }

    let(config_mostly_default_randomized) {
      Ai4cr::NeuralNetwork::Cmn::MiniNet.config_rand(
        height: inputs_size, width: outputs_size, history_size: 10
      )
    }
    let(config_mnm_prelu) { config_mostly_default_randomized.merge(learning_style: Ai4cr::NeuralNetwork::LS_PRELU) }
    let(config_mnm_relu) { config_mostly_default_randomized.merge(learning_style: Ai4cr::NeuralNetwork::LS_RELU) }
    let(config_mnm_sigmoid) { config_mostly_default_randomized.merge(learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID) }
    let(config_mnm_tanh) { config_mostly_default_randomized.merge(learning_style: Ai4cr::NeuralNetwork::LS_TANH) }

    let(mnm_prelu) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }
    let(mnm_relu) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }
    let(mnm_sigmoid) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }
    let(mnm_tanh) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }

    describe "train_and_and_cross_breed_team_set_using_sequence" do
      let(qty_new_members) { 10 }
      let(learning_rates) { (1..qty_new_members).map { |i| i / 11.0 } }
      let(momentums) { learning_rates.map { |i| i / 2 + 0.25 } }
      let(bias_defaults) { learning_rates.map { |i| i / 1.5 } }
      let(expected_second_and_third_lower) { [true, true, true] }

      context "successive generations score better (i.e.: lower errors) for" do
        it "Ai4cr::NeuralNetwork::LS_PRELU" do
          first_gen_members = mnm_prelu.build_team(qty_new_members, **config_mnm_prelu)
          first_gen_members.each_with_index do |member, i|
            member.learning_rate = learning_rates[i]
            member.momentum = momentums[i]
            member.bias_default = bias_defaults[i]
          end
          first_gen_members_clones = first_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

          resulting_teams_with_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_prelu, first_gen_members, inputs_sequence_train, outputs_sequence_train)
          resulting_teams_without_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_prelu, first_gen_members_clones, inputs_sequence_train, outputs_sequence_train, false)

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_with_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_prelu,
            resulting_teams_with_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_without_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_prelu,
            resulting_teams_without_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          expect(resulting_teams_with_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
          expect(resulting_teams_without_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
        end

        it "Ai4cr::NeuralNetwork::LS_RELU" do
          first_gen_members = mnm_relu.build_team(qty_new_members, **config_mnm_relu)
          first_gen_members.each_with_index do |member, i|
            member.learning_rate = learning_rates[i]
            member.momentum = momentums[i]
            member.bias_default = bias_defaults[i]
          end
          first_gen_members_clones = first_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

          resulting_teams_with_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_relu, first_gen_members, inputs_sequence_train, outputs_sequence_train)
          resulting_teams_without_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_prelu, first_gen_members_clones, inputs_sequence_train, outputs_sequence_train, false)

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_with_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_relu,
            resulting_teams_with_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_without_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_relu,
            resulting_teams_without_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          expect(resulting_teams_with_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
          expect(resulting_teams_without_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
        end

        it "Ai4cr::NeuralNetwork::LS_SIGMOID" do
          first_gen_members = mnm_sigmoid.build_team(qty_new_members, **config_mnm_sigmoid)
          first_gen_members.each_with_index do |member, i|
            member.learning_rate = learning_rates[i]
            member.momentum = momentums[i]
            member.bias_default = bias_defaults[i]
          end
          first_gen_members_clones = first_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

          resulting_teams_with_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_sigmoid, first_gen_members, inputs_sequence_train, outputs_sequence_train)
          resulting_teams_without_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_prelu, first_gen_members_clones, inputs_sequence_train, outputs_sequence_train, false)

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_with_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_sigmoid,
            resulting_teams_with_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_without_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_sigmoid,
            resulting_teams_without_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          expect(resulting_teams_with_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
          expect(resulting_teams_without_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
        end
        it "Ai4cr::NeuralNetwork::LS_TANH" do
          first_gen_members = mnm_tanh.build_team(qty_new_members, **config_mnm_tanh)
          first_gen_members.each_with_index do |member, i|
            member.learning_rate = learning_rates[i]
            member.momentum = momentums[i]
            member.bias_default = bias_defaults[i]
          end
          first_gen_members_clones = first_gen_members.map { |member| Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(member.to_json) }

          resulting_teams_with_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_tanh, first_gen_members, inputs_sequence_train, outputs_sequence_train)
          resulting_teams_without_cross_breeding = train_and_and_cross_breed_team_set_using_sequence(mnm_prelu, first_gen_members_clones, inputs_sequence_train, outputs_sequence_train, false)

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_with_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_tanh,
            resulting_teams_with_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          puts_debug
          puts_debug "-"*10
          puts_debug "resulting_teams_without_cross_breeding guesses:"
          eval_team_set_using_sequence(
            mnm_tanh,
            resulting_teams_without_cross_breeding[:members_per_generation],
            inputs_sequence_for_guessing, outputs_sequence_expected
          )

          expect(resulting_teams_with_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
          expect(resulting_teams_without_cross_breeding[:is_error_decreasing]).to eq(expected_second_and_third_lower)
        end
      end
    end
  end
end
