require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::Rnn do
  describe "when importing and exporting as JSON" do
    rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new
    rnn_to_json = rnn.to_json
    rnn_from_json = Ai4cr::NeuralNetwork::Cmn::Rnn.from_json(rnn_to_json)
    re_exported_json = rnn_from_json.to_json

    # puts rnn_from_json.pretty_inspect

    context "when comparing exported vs imported matches for" do
      it "config" do
        re_exported_json["config"].should eq(rnn_to_json["config"])
      end

      it "layer_index_max" do
        re_exported_json["layer_index_max"].should eq(rnn_to_json["layer_index_max"])
      end

      it "layer_range" do
        re_exported_json["layer_range"].should eq(rnn_to_json["layer_range"])
      end

      it "time_col_index_max" do
        re_exported_json["time_col_index_max"].should eq(rnn_to_json["time_col_index_max"])
      end

      it "time_col_range" do
        re_exported_json["time_col_range"].should eq(rnn_to_json["time_col_range"])
      end

      it "mini_net_configs" do
        re_exported_json["mini_net_configs"].should eq(rnn_to_json["mini_net_configs"])
      end

      it "mini_net_set" do
        re_exported_json["mini_net_set"].should eq(rnn_to_json["mini_net_set"])
      end
    end
  end

  describe "#eval" do
    time_col_qty = 3
    config = Ai4cr::NeuralNetwork::Cmn::RnnConcerns::Config.new(
      input_state_size: 11, hidden_state_size: 22, output_state_size: 11,
      time_col_qty: time_col_qty
    )
    rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)

    simple_wave_rise = (0..10).to_a.map { |i| (0..10).to_a.map { |j| i == j ? 1.0 : 0.0 } }
    training_data = simple_wave_rise + simple_wave_rise.reverse + simple_wave_rise + simple_wave_rise.reverse

    it "changes the values of outputs_guessed" do
      # puts "\nBEFORE:\n"
      # puts rnn.to_json.pretty_inspect

      outputs_guessed_before = rnn.outputs_guessed.clone

      offset = 0
      time_col_from = offset
      time_col_to = offset + time_col_qty - 1
      rnn.eval(training_data[time_col_from..time_col_to])
      (rnn.outputs_guessed).should_not eq(outputs_guessed_before)

      # puts "\nAFTER:\n"
      # puts rnn.to_json.pretty_inspect
      # puts "\nrnn.outputs_guessed:\n"
      # puts rnn.outputs_guessed
      puts "\nrnn.guesses_best:\n"
      puts rnn.guesses_best
    end
  end

  describe "#train" do
    # TODO
    # # train
    # training_data_size = training_data.size
    # time_col_qty

    time_col_qty = 3
    config = Ai4cr::NeuralNetwork::Cmn::RnnConcerns::Config.new(
      input_state_size: 11, hidden_state_size: 22, output_state_size: 11,
      time_col_qty: time_col_qty
    )

    simple_wave_rise = (0..10).to_a.map { |i| (0..10).to_a.map { |j| i == j ? 1.0 : 0.0 } }
    training_data = simple_wave_rise + simple_wave_rise.reverse + simple_wave_rise + simple_wave_rise.reverse

    it "changes the values of input_deltas" do
      rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)
      # puts "\nBEFORE:\n"
      # puts rnn.to_json.pretty_inspect

      input_deltas_before = rnn.input_deltas.clone
      puts "\n input_deltas_before: \n"
      puts input_deltas_before

      # offset = 0
      # time_col_from = offset
      # time_col_to = offset + time_col_qty - 1
      # rnn.train(training_data[time_col_from..time_col_to])
      # (rnn.input_deltas).should_not eq(input_deltas_before)

      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.input_deltas).should_not eq(input_deltas_before)

      # puts "\nAFTER:\n"
      # puts rnn.to_json.pretty_inspect
      puts "\n rnn.input_deltas: \n"
      puts rnn.input_deltas
    end

    it "changes the values of error_total" do
      rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)
      # puts "\nBEFORE:\n"
      # puts rnn.to_json.pretty_inspect

      error_total_before = rnn.error_total.clone
      puts "\n error_total_before: \n"
      puts error_total_before

      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.error_total).should_not eq(error_total_before)

      # puts "\nAFTER:\n"
      # puts rnn.to_json.pretty_inspect
      puts "\n rnn.error_total: \n"
      puts rnn.error_total
    end
  end

  describe "#train (twice)" do
    # TODO
    # # train
    # training_data_size = training_data.size
    # time_col_qty

    time_col_qty = 3
    config = Ai4cr::NeuralNetwork::Cmn::RnnConcerns::Config.new(
      input_state_size: 11, hidden_state_size: 22, output_state_size: 11,
      time_col_qty: time_col_qty
    )

    simple_wave_rise = (0..10).to_a.map { |i| (0..10).to_a.map { |j| i == j ? 1.0 : 0.0 } }
    training_data = simple_wave_rise + simple_wave_rise.reverse + simple_wave_rise + simple_wave_rise.reverse

    it "changes the values of input_deltas" do
      rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)
      # puts "\nBEFORE:\n"
      # puts rnn.to_json.pretty_inspect

      input_deltas_before = rnn.input_deltas.clone
      puts "\n input_deltas_before: \n"
      puts input_deltas_before

      # offset = 0
      # time_col_from = offset
      # time_col_to = offset + time_col_qty - 1
      # rnn.train(training_data[time_col_from..time_col_to])
      # (rnn.input_deltas).should_not eq(input_deltas_before)

      # first time
      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.input_deltas).should_not eq(input_deltas_before)

      input_deltas_before2 = rnn.input_deltas.clone
      puts "\n input_deltas_before2: \n"
      puts input_deltas_before2

      # second time
      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.input_deltas).should_not eq(input_deltas_before)
      (rnn.input_deltas).should_not eq(input_deltas_before2)

      # puts "\nAFTER:\n"
      # puts rnn.to_json.pretty_inspect
      puts "\n input_deltas_before: \n"
      puts input_deltas_before
      puts "\n input_deltas_before2: \n"
      puts input_deltas_before2
      puts "\n rnn.input_deltas: \n"
      puts rnn.input_deltas
    end

    it "changes the values of error_total" do
      rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new(config)
      # puts "\nBEFORE:\n"
      # puts rnn.to_json.pretty_inspect

      error_total_before = rnn.error_total.clone
      puts "\n error_total_before: \n"
      puts error_total_before

      # first time
      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.error_total).should_not eq(error_total_before)

      error_total_before2 = rnn.error_total.clone
      puts "\n error_total_before2: \n"
      puts error_total_before2

      # second time
      offset = 0
      input_time_col_from = offset
      input_time_col_to = offset + time_col_qty - 1

      output_time_col_from = input_time_col_to + 1
      output_time_col_to = output_time_col_from + time_col_qty - 1

      # time_col_to = offset + time_col_qty - 1
      rnn.train(training_data[input_time_col_from..input_time_col_to], training_data[output_time_col_from..output_time_col_to])
      (rnn.error_total).should_not eq(error_total_before)
      (rnn.error_total).should_not eq(error_total_before2)

      # puts "\nAFTER:\n"
      # puts rnn.to_json.pretty_inspect
      puts "\n error_total_before: \n"
      puts error_total_before
      puts "\n error_total_before2: \n"
      puts error_total_before2
      puts "\n rnn.error_total: \n"
      puts rnn.error_total
    end
  end
end
