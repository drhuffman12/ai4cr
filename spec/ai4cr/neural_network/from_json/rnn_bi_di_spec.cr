require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnBiDi do
  context "correctly exports and imports" do
    let(orig) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new } # (input_size: input_size, output_size: output_size) }

    let(input_set_given) {
      [
        [0.1, 0.2],
        [0.3, 0.4],
      ]
    }
    let(output_set_expected) { [[0.4], [0.6]] }

    context "correctly exports and imports" do
      pending "the whole object" do
        # NOTE: Due to rounding errors of Float64 values during import and export of JSON, this test might fail; just re-run.
        # NOTE: For now, mark as 'pending', but ...
        #   There are float rounding discrepancies between to/from json values.

        a = JSON.parse(orig.to_json)

        a_copy = orig.class.from_json(orig.to_json)
        b = JSON.parse(a_copy.to_json)

        assert_approximate_equality_of_nested_list(b, a, 1.0e-15)
      end

      pending "the whole object after training once" do
        # NOTE: Due to rounding errors of Float64 values during import and export of JSON, this test might fail; just re-run.

        orig.train(input_set_given, output_set_expected)

        a = JSON.parse(orig.to_json)

        a_copy = orig.class.from_json(orig.to_json)
        b = JSON.parse(a_copy.to_json)

        assert_approximate_equality_of_nested_list(b, a, 1.0e-15)
      end

      pending "the whole object after training 10 times" do
        # NOTE: Due to rounding errors of Float64 values during import and export of JSON, this test might fail; just re-run.

        n = 10
        n.times { orig.train(input_set_given, output_set_expected) }

        a = JSON.parse(orig.to_json)

        a_copy = orig.class.from_json(orig.to_json)
        b = JSON.parse(a_copy.to_json)

        assert_approximate_equality_of_nested_list(b, a, 1.0e-15)
      end
    end
  end
end
