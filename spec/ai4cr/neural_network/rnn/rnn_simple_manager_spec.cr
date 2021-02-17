require "./../../../spec_helper"
require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new }
  let(delta_child_1) { Ai4cr::Data::Utils.rand_neg_half_to_pos_one_and_half_no_zero_no_one }
  let(delta_child_2) { Ai4cr::Data::Utils.rand_neg_half_to_pos_one_and_half_no_zero_no_one }

  let(ancestor_adam_value) { 0.1 }
  let(ancestor_eve_value) { 0.9 }
  let(expected_child_1_value) { ancestor_adam_value + delta_child_1 * (ancestor_eve_value - ancestor_adam_value) }

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

  let(child_1) {
    # cain
    child = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
    child.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"
    child
  }

  context "For Adam and Eve examples" do
    before_each do
      my_breed_manager.counter_reset
    end

    describe "#breed" do
      let(ancestor_1) { my_breed_manager.create(name: "defaults 1") }
      let(ancestor_2) { my_breed_manager.create(name: "defaults 2") }

      let(ancestor_3) { my_breed_manager.create(name: "non-default history_size", history_size: Ai4cr::NeuralNetwork::Rnn::RnnSimple::HISTORY_SIZE_DEFAULT + 1 + rand(2)) }
      let(ancestor_4) { my_breed_manager.create(name: "non-default io_offset", io_offset: Ai4cr::NeuralNetwork::Rnn::RnnSimple::IO_OFFSET_DEFAULT + 1 + rand(2)) }
      let(ancestor_5) { my_breed_manager.create(name: "non-default time_col_qty", time_col_qty: Ai4cr::NeuralNetwork::Rnn::RnnSimple::TIME_COL_QTY_MIN + 1 + rand(2)) }
      let(ancestor_6) { my_breed_manager.create(name: "non-default input_size", input_size: Ai4cr::NeuralNetwork::Rnn::RnnSimple::INPUT_SIZE_MIN + 1 + rand(2)) }
      let(ancestor_7) { my_breed_manager.create(name: "non-default output_size", output_size: Ai4cr::NeuralNetwork::Rnn::RnnSimple::OUTPUT_SIZE_MIN + 1 + rand(2)) }
      let(ancestor_8) { my_breed_manager.create(name: "non-default hidden_layer_qty", hidden_layer_qty: Ai4cr::NeuralNetwork::Rnn::RnnSimple::HIDDEN_LAYER_QTY_MIN + 1 + rand(2)) }
      let(ancestor_9) { my_breed_manager.create(name: "non-default hidden_size_given", hidden_size_given: 2 + rand(2)) }
      let(ancestor_10) { my_breed_manager.create(name: "non-default bias_disabled", bias_disabled: true) }
      let(ancestor_11) { my_breed_manager.create(name: "non-default learning_style", learning_style: (LearningStyle.values - [Ai4cr::NeuralNetwork::Rnn::RnnSimple::LEARNING_STYLE_DEFAULT]).sample) }

      context "when parents have same structure params values" do
        it "does NOT raise" do
          ancestor_a = ancestor_1
          ancestor_b = ancestor_2

          expect(ancestor_a.history_size).to eq(ancestor_b.history_size)
          expect(ancestor_a.io_offset).to eq(ancestor_b.io_offset)
          expect(ancestor_a.time_col_qty).to eq(ancestor_b.time_col_qty)
          expect(ancestor_a.input_size).to eq(ancestor_b.input_size)
          expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
          expect(ancestor_a.hidden_layer_qty).to eq(ancestor_b.hidden_layer_qty)
          expect(ancestor_a.hidden_size_given).to eq(ancestor_b.hidden_size_given)
          expect(ancestor_a.bias_disabled).to eq(ancestor_b.bias_disabled)
          expect(ancestor_a.bias_default).to eq(ancestor_b.bias_default)
          expect(ancestor_a.learning_style).to eq(ancestor_b.learning_style)

          expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.not_to raise_error
        end
      end

      context "when parents have differing structure param values for" do
        context "io_offset" do
          it "raises" do
            ancestor_a = ancestor_1
            ancestor_b = ancestor_3

            expect(ancestor_a.history_size).not_to eq(ancestor_b.history_size)

            expect(ancestor_a.io_offset).to eq(ancestor_b.io_offset)
            expect(ancestor_a.time_col_qty).to eq(ancestor_b.time_col_qty)
            expect(ancestor_a.input_size).to eq(ancestor_b.input_size)
            expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
            expect(ancestor_a.hidden_layer_qty).to eq(ancestor_b.hidden_layer_qty)
            expect(ancestor_a.hidden_size_given).to eq(ancestor_b.hidden_size_given)
            expect(ancestor_a.bias_disabled).to eq(ancestor_b.bias_disabled)
            expect(ancestor_a.bias_default).to eq(ancestor_b.bias_default)
            expect(ancestor_a.learning_style).to eq(ancestor_b.learning_style)

            expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error(Ai4cr::Breed::StructureError)
          end
        end
      end

      context "children have expected values for" do
        # let(child_1) {
        #   # cain
        #   child = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
        #   child.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"
        #   child
        # }

        let(ancestor_adam_json) { JSON.parse(ancestor_adam.to_json) }
        let(ancestor_eve_json) { JSON.parse(ancestor_eve.to_json) }
        let(child_1_json) { JSON.parse(child_1.to_json) }

        describe "#to_json" do
          context "for a new MiniNetManager with two initial 'ancestors' and one 'child'" do
            context "correctly exports" do
              it "the whole initial object" do
                expect(ancestor_adam).not_to be_nil
                expect(ancestor_eve).not_to be_nil
                expect(child_1).not_to be_nil

                # my_breed_manager.counter.inc("foo")
                counter = my_breed_manager.counter
                puts_debug
                puts_debug "counter.to_json: #{counter.to_json}"
                puts_debug
                puts_debug "my_breed_manager.to_json: #{my_breed_manager.to_json}"
                puts_debug

                # NOTE: 'exported' vs 'expected'
                exported_json = my_breed_manager.to_json
                expected_json = "{\"mini_net_manager\":{}}"

                expect(exported_json).to be_a(String)
                expect(exported_json).to eq(expected_json)
              end
            end
          end
        end

        describe "#parts_to_copy stay the same, such as" do
          it "misc instance variables" do
            puts_debug
            puts_debug "  ancestor_adam.to_json: #{ancestor_adam.to_json}"
            puts_debug "  ancestor_eve.to_json: #{ancestor_eve.to_json}"
            puts_debug "  child_1.to_json: #{child_1.to_json}"
            puts_debug

            # "history_size",
            [
              "io_offset", "time_col_qty", "input_size", "output_size",
              "hidden_layer_qty", "hidden_size_given", "bias_disabled", "learning_style",
            ].each do |var|
              puts_debug
              puts_debug "var: #{var}"
              puts_debug
              expect(ancestor_adam_json[var]).to eq(ancestor_eve_json[var])
              expect(child_1_json[var]).to eq(ancestor_adam_json[var])
            end
          end
        end

        describe "#mix_parts are delta-mixed, such as" do
          it "bias_default" do
            ancestor_adam_value = ancestor_adam.bias_default
            ancestor_eve_value = ancestor_eve.bias_default
            expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

            expect(ancestor_adam.bias_default).not_to eq(ancestor_eve.bias_default)
            expect(child_1.bias_default).to eq(expected_child_1_value)
          end

          it "learning_rate" do
            ancestor_adam_value = ancestor_adam.learning_rate
            ancestor_eve_value = ancestor_eve.learning_rate
            expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

            expect(ancestor_adam.learning_rate).not_to eq(ancestor_eve.learning_rate)
            expect(child_1.learning_rate).to eq(expected_child_1_value)
          end

          it "momentum" do
            ancestor_adam_value = ancestor_adam.momentum
            ancestor_eve_value = ancestor_eve.momentum
            expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

            expect(ancestor_adam.momentum).not_to eq(ancestor_eve.momentum)
            expect(child_1.momentum).to eq(expected_child_1_value)
          end

          it "deriv_scale" do
            ancestor_adam_value = ancestor_adam.deriv_scale
            ancestor_eve_value = ancestor_eve.deriv_scale
            expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

            expect(ancestor_adam.deriv_scale).not_to eq(ancestor_eve.deriv_scale)
            expect(child_1.deriv_scale).to eq(expected_child_1_value)
          end
        end

        context "NOT copied and NOT mixed, but instead freshly initialized" do
          context "error_stats" do
            it "history_size" do
              ancestor_adam_value = ancestor_adam.error_stats.history_size
              ancestor_eve_value = ancestor_eve.error_stats.history_size
              # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
              expected_child_1_value = ancestor_adam_value

              expect(ancestor_adam_value).to eq(ancestor_eve_value)
              expect(child_1.error_stats.history_size).to eq(expected_child_1_value)
            end

            it "distance" do
              # TODO
              ancestor_adam_value = ancestor_adam.error_stats.distance
              ancestor_eve_value = ancestor_eve.error_stats.distance
              # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
              expected_child_1_value = -1.0

              expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
              expect(child_1.error_stats.distance).to eq(expected_child_1_value)
            end

            it "history.size" do
              ancestor_adam_value = ancestor_adam.error_stats.history.size
              ancestor_eve_value = ancestor_eve.error_stats.history.size
              # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
              expected_child_1_value = 0

              expect(ancestor_adam_value).to eq(ancestor_eve_value)
              expect(child_1.error_stats.history.size).to eq(expected_child_1_value)
            end

            it "score" do
              # TODO
              ancestor_adam_value = ancestor_adam.error_stats.score
              ancestor_eve_value = ancestor_eve.error_stats.score
              # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
              expected_child_1_value = 1.8446744073709552e+19

              expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
              expect(child_1.error_stats.score).to eq(expected_child_1_value)
            end
          end
        end
      end
    end

    context "when both parents are trained, then breed, and then all train" do
      # it "when they have only 1 child, the child has smaller error_stats.distance" do
      #   # cain
      #   child = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
      #   child.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"
      #   child

      #   ancestor_adam.train(inputs, outputs)
      #   ancestor_eve.train(inputs, outputs)
      #   child_1.train(inputs, outputs)

      #   puts_debug
      #   puts_debug "ancestor_adam.class: #{ancestor_adam.class}"
      #   puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
      #   puts_debug
      #   puts_debug "ancestor_eve.class: #{ancestor_eve.class}"
      #   puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
      #   puts_debug
      #   puts_debug "child_1.class: #{child_1.class}"
      #   puts_debug "child_1: #{child_1.to_json}"
      #   puts_debug
      #   expect(child_1.error_stats.distance).to be < (ancestor_adam.error_stats.distance)
      #   expect(child_1.error_stats.distance).to be < (ancestor_eve.error_stats.distance)

      # end

      it "when they have N children, at least two of the children have smaller error_stats.distance" do
        children_qty = 10

        training_rounds = 3.times.to_a
        training_rounds.each do
          ancestor_adam.train(inputs, outputs)
          ancestor_eve.train(inputs, outputs)
        end

        children = children_qty.times.to_a.map do |i|
          child = my_breed_manager.breed(ancestor_adam, ancestor_eve)
          child.name = "Child ##{i} of #{ancestor_adam.name} and #{ancestor_eve.name}"
          child
        end

        training_rounds.each do
          ancestor_adam.train(inputs, outputs)
          ancestor_eve.train(inputs, outputs)
          children.each do |child|
            child.train(inputs, outputs)
          end
        end

        qty_better = children.map do |child|
          puts_debug
          puts_debug "ancestor_adam.name: #{child.name}, error_stats.distance: #{ancestor_adam.error_stats.distance}"
          puts_debug "ancestor_eve.name: #{child.name}, error_stats.distance: #{ancestor_eve.error_stats.distance}"
          puts_debug "child.name: #{child.name}, error_stats.distance: #{child.error_stats.distance}"
          puts_debug
          # expect(child.error_stats.distance).to be < (ancestor_adam.error_stats.distance)
          # expect(child.error_stats.distance).to be < (ancestor_eve.error_stats.distance)
          is_better = child.error_stats.distance < ancestor_adam.error_stats.distance &&
                      child.error_stats.distance < ancestor_eve.error_stats.distance

          puts_debug "is_better: #{is_better}"

          is_better ? 1 : 0
        end.sum
        puts_debug "children.size: #{children.size}"
        puts_debug "qty_better: #{qty_better}"
        puts_debug

        expect(qty_better).to be >= 2
      end
    end
  end

  describe "#build_team" do
    context "when using all defaults" do
      it "creates specified quantity of members" do
        qty_new_members = 4
        params = Ai4cr::NeuralNetwork::Rnn::RnnSimple.new.config
        next_gen_members = my_breed_manager.build_team(qty_new_members, **params)
        expect(next_gen_members.size).to eq(qty_new_members)
      end
    end
  end

  describe "#train_team" do
    it "successive generations score better (i.e.: lower errors)" do
      max_members = 5
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
      max_members = 5
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
