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

      # let(time_col_qty) { 4 }
      # let(hidden_layer_qty) { 1 }

      # first_gen_members_scored: 3.171080497616357e+30
      # 386 => ▴▴▴▴▴▴▴▴▴▴ @ 2.9962787852077016e+21
      # 382 => ▴▴▴▴▴▴▴▴▴▴ @ 4.321539274778076e+21
      # 383 => ▴▴▴▴▴▴▴▴▴▴ @ 3.565246009683592e+23
      # 387 => ▴▴▴▴▴▴▴▴▴▴ @ 3.0336848935362745e+29
      # 385 => ▴▴▴▴▴▴▴▴▴▴ @ 2.3155511010525606e+25
      # 384 => ▴▴▴▴▴▴▴▴▴▴ @ 1.6210743292589577e+30
      # 388 => ▴▴▴▴▴▴▴▴▴▴ @ 2.7318190668835828e+25
      # 381 => ▴▴▴▴▴▴▴▴▴▴ @ 3.6272299887002884e+26
      # 389 => ▴▴▴▴▴▴▴▴▴▴ @ 2.9674479095739515e+31
      # 390 => ▴▴▴▴▴▴▴▴▴▴ @ 1.1146950126849715e+29

      # second_gen_members_scored: 1.5966902119228623e+18
      # 460 => ▴▴▴▴▴▴▴▴▴▴ @ 1.1408180781164006e+18
      # 411 => ▴▴▴▴▴▴▴▴▴▴ @ 2.4649604389660698e+17
      # 430 => ▴▴▴▴▴▴▴▴▴▴ @ 9.667075547466113e+18
      # 391 => ▴▴▴▴▴▴▴▴▴▴ @ 1.3655909747618217e+18
      # 480 => ▴▴▴▴▴▴▴▴▴▴ @ 1.996796074020602e+18
      # 474 => ▴▴▴▴▴▴▴▴▴▴ @ 6.221118834020808e+17
      # 470 => ▴▴▴▴▴▴▴▴▴▴ @ 2.4499257092119398e+17
      # 475 => ▴▴▴▴▴▴▴▴▴▴ @ 2.1908468299125427e+17
      # 468 => ▴▴▴▴▴▴▴▴▴▴ @ 1.8058984589372483e+17
      # 413 => ▴▴▴▴▴▴▴▴▴▴ @ 2.8334641775882714e+17

      # third_gen_members_scored: 2.171687810154346e+16
      # 553 => ▴▴▴▴▴▴▴▴▴▴ @ 4.319007968195005e+15
      # 515 => ▴▴▴▴▴▴▴▴▴▴ @ 4.519837484733632e+15
      # 523 => ▴▴▴▴▴▴▴▴▴▴ @ 1.622686714868059e+16
      # 548 => ▴▴▴▴▴▴▴▴▴▴ @ 1.628037498045222e+16
      # 536 => ▴▴▴▴▴▴▴▴▴▴ @ 1.8847057437670416e+16
      # 495 => ▴▴▴▴▴▴▴▴▴▴ @ 2.4837082889173624e+16
      # 518 => ▴▴▴▴▴▴▴▴▴▴ @ 3.1761441879799064e+16
      # 552 => ▴▴▴▴▴▴▴▴▴▴ @ 3.1821109197211916e+16
      # 556 => ▴▴▴▴▴▴▴▴▴▴ @ 3.2152684043093496e+16
      # 497 => ▴▴▴▴▴▴▴▴▴▴ @ 3.640331798642464e+16

      let(time_col_qty) { 16 }
      let(hidden_layer_qty) { 4 }

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
