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
    context "when using an arbitrary set of float values for io" do
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
        puts "#train_team_using_sequence (arbitrary set of float values):"
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

    context "when using a text file as io data" do
      let(file_path) { "./spec_bench/support/neural_network/data/eng-web_002_GEN_01_read.txt" }
      # let(float_bits_from_file) { Ai4cr::Utils::Rand.text_file_to_fiod(file_path) }

      let(time_col_qty) { 4 }
      let(hidden_layer_qty) { 1 }

      # # e.g.:
      # first_gen_members_scored: 3.171080497616357e+30
      # 386 => ▴▴▴▴▴▴▴▴▴▴ @ 2.9962787852077016e+21
      # ...
      # 390 => ▴▴▴▴▴▴▴▴▴▴ @ 1.1146950126849715e+29

      # second_gen_members_scored: 1.5966902119228623e+18
      # 460 => ▴▴▴▴▴▴▴▴▴▴ @ 1.1408180781164006e+18
      # ...
      # 413 => ▴▴▴▴▴▴▴▴▴▴ @ 2.8334641775882714e+17

      # third_gen_members_scored: 2.171687810154346e+16
      # 553 => ▴▴▴▴▴▴▴▴▴▴ @ 4.319007968195005e+15
      # ...
      # 497 => ▴▴▴▴▴▴▴▴▴▴ @ 3.640331798642464e+16

      # # e.g.:
      # first_gen_members_scored: 11272500.728510443
      # 388 => ▴▴▴▴▴▴▴▴▴▴ @ 27159480.43494436
      # ...
      # 390 => ▴▴▴▴▴▴▴▴▴▴ @ 3404231.491292608

      # second_gen_members_scored: 1322158.0917929376
      # 452 => ▴▴▴▴▴▴▴▴▴▴ @ 712847.2277136235
      # ...
      # 383 => ▴▴▴▴▴▴▴▴▴▴ @ 7928651.50162725

      # third_gen_members_scored: 65458.56929389067
      # 499 => ▴▴▴▴▴▴▴▴▴▴ @ 12777.101255848469
      # ...
      # 559 => ▴▴▴▴▴▴▴▴▴▴ @ 115833.22024213054

      # let(time_col_qty) { 16 }
      # let(hidden_layer_qty) { 4 }
      # # from ???
      # # /home/drhuffman/.crenv/versions/0.36.0/share/crystal/src/primitives.cr:255:3 in 'run'
      # #   from ???
      # # src/ai4cr/breed/manager.cr:105:17 in 'breed'
      # #   from src/ai4cr/breed/manager.cr:269:30 in '->'
      # #   from ???src/ai4cr/breed/manager.cr:105:17 in 'breed'

      let(io_offset) { time_col_qty }

      let(file_path) { "./spec_bench/support/neural_network/data/eng-web_002_GEN_01_read.txt" }
      let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
      let(file_type_iod) { Ai4cr::Utils::IoData::FileType::Iod }
      let(prefix_raw_qty) { time_col_qty }
      let(prefix_raw_char) { " " }

      let(io_set_text_file) do
        Ai4cr::Utils::IoData::TextFile.new(
          file_path, file_type_raw,
          prefix_raw_qty, prefix_raw_char
        )
      end

      let(raw) { io_set_text_file.raw }
      let(iod) { io_set_text_file.iod }

      let(ios) { io_set_text_file.iod_to_io_set_with_offset_time_cols(time_col_qty, io_offset) }
      # let(input_set) { ios[:input_set] }
      # let(output_set) { ios[:output_set] }
      let(inputs_sequence) { ios[:input_set] }
      let(outputs_sequence) { ios[:output_set] }

      it "successive generations score better (i.e.: lower errors)" do
        # TODO: (a) move to 'spec_bench' and (b) replace here with more 'always' tests
        max_members = 10
        qty_new_members = max_members

        params = Ai4cr::NeuralNetwork::Rnn::RnnSimple.new(
          io_offset: io_offset,
          time_col_qty: time_col_qty,
          input_size: inputs_sequence.first.first.size,
          output_size: outputs_sequence.first.first.size,
          hidden_layer_qty: hidden_layer_qty,
        ).config

        first_gen_members = my_breed_manager.build_team(qty_new_members, **params)

        # puts
        # puts "first_gen_members: #{first_gen_members}"
        puts "inputs_sequence.size: #{inputs_sequence.size}"
        puts "inputs_sequence.first.size: #{inputs_sequence.first.size}"
        puts "inputs_sequence.first.first.size: #{inputs_sequence.first.first.size}"
        puts "inputs_sequence.class: #{inputs_sequence.class}"
        puts "outputs_sequence.class: #{outputs_sequence.class}"
        puts "params: #{params}"

        second_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, first_gen_members, max_members)
        third_gen_members = my_breed_manager.train_team_using_sequence(inputs_sequence, outputs_sequence, second_gen_members, max_members)
        first_gen_members_scored = first_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
        first_gen_members_stats = first_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

        second_gen_members_scored = second_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
        second_gen_members_stats = second_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

        third_gen_members_scored = third_gen_members.map { |member| member.error_stats.score }.sum / qty_new_members
        third_gen_members_stats = third_gen_members.map { |member| "#{member.birth_id} => #{member.error_stats.plot_error_distance_history} @ #{member.error_stats.score}" }

        puts
        puts "#train_team_using_sequence (text from Bible):"
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
end
