require "./../../../spec_helper"
require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
# describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
  # subject { Ai4cr::NeuralNetwork::Cmn::RnnSimple }
  describe "#initialize" do
    context "when NOT passing in any values" do
      # let(rnn_simple) { subject.new }
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }
      # subject { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }
      # rnn_simple = Ai4cr::NeuralNetwork::Cmn::RnnSimple.new

      # puts "rnn_simple: #{rnn_simple.pretty_inspect}"
      # puts "rnn_simple.node_input_sizes.first: #{rnn_simple.node_input_sizes.first.pretty_inspect}"
      # puts "rnn_simple.node_input_sizes[1]: #{rnn_simple.node_input_sizes[1].pretty_inspect}"
      # puts "rnn_simple.node_input_sizes.last: #{rnn_simple.node_input_sizes.last.pretty_inspect}"

      it "just some debugging" do
        puts rnn_simple.to_pretty_json

        rnn_simple.layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            # debug_info = {"li": li, "ti": ti, "rnn_simple.node_input_sizes[li][ti]": rnn_simple.node_input_sizes[li][ti]}
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end
      it "has no errors" do
        expect(rnn_simple.errors.empty?).to be_true
        expect(rnn_simple.errors.is_a?(Hash(Symbol, String))).to be_true
        expect(rnn_simple.errors).to eq(Hash(Symbol, String).new)
      end

      it "is valid" do
        expect(rnn_simple.valid?).to be_true
      end

      context "has expected value for property" do
        it "time_col_qty" do
          expect(rnn_simple.time_col_qty).to eq(2)
        end

        it "hidden_layer_qty" do
          expect(rnn_simple.hidden_layer_qty).to eq(1)
        end

        it "input_size" do
          expect(rnn_simple.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_simple.output_size).to eq(1)
        end

        it "hidden_size" do
          expect(rnn_simple.hidden_size).to eq(3)
        end
      end

    end

    # # context "when passing in valid values" do
    # #   # let(rnn_simple) { subject.new }
    # #   time_col_qty = 3
    # #   hidden_layer_qty = 2
    # #   input_size = 3
    # #   output_size = 2
    # #   hidden_size_given = 4

    # #   rnn_simple = Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
    # #     time_col_qty,
    # #     hidden_layer_qty,
    # #     input_size,
    # #     output_size,
    # #     hidden_size_given
    # #   )

    # #   it "has no errors" do
    # #     rnn_simple.errors.empty?.should be_true
    # #     rnn_simple.errors.is_a?(Hash(Symbol, String)).should be_true
    # #     rnn_simple.errors.should eq(Hash(Symbol, String).new)
    # #   end

    # #   it "is valid" do
    # #     rnn_simple.valid?.should be_true
    # #   end

    # #   context "has expected value for property" do
    # #     it "time_col_qty" do
    # #       rnn_simple.time_col_qty.should eq(time_col_qty)
    # #     end

    # #     it "hidden_layer_qty" do
    # #       rnn_simple.hidden_layer_qty.should eq(hidden_layer_qty)
    # #     end

    # #     it "input_size" do
    # #       rnn_simple.input_size.should eq(input_size)
    # #     end

    # #     it "output_size" do
    # #       rnn_simple.output_size.should eq(output_size)
    # #     end

    # #     it "hidden_size" do
    # #       rnn_simple.hidden_size.should eq(hidden_size_given)
    # #     end
    # #   end
    # # end

    # # context "when passing in invalid values (too small hidden_size_given)" do
    # #   # let(rnn_simple) { subject.new }
    # #   time_col_qty = 3
    # #   hidden_layer_qty = 2
    # #   input_size = 3
    # #   output_size = 2
    # #   hidden_size_given = 1

    # #   rnn_simple = Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
    # #     time_col_qty,
    # #     hidden_layer_qty,
    # #     input_size,
    # #     output_size,
    # #     hidden_size_given
    # #   )

    # #   it "has errors" do
    # #     rnn_simple.errors.empty?.should be_false
    # #     rnn_simple.errors.is_a?(Hash(Symbol, String)).should be_true
    # #     rnn_simple.errors.should eq({:hidden_size_given => "hidden_size_given must be at least 3 if supplied (otherwise it defaults to sum of @input_size and @output_size"})
    # #   end

    # #   it "is valid" do
    # #     rnn_simple.valid?.should be_false
    # #   end

    # #   context "has expected value for property" do
    # #     it "time_col_qty" do
    # #       rnn_simple.time_col_qty.should eq(time_col_qty)
    # #     end

    # #     it "hidden_layer_qty" do
    # #       rnn_simple.hidden_layer_qty.should eq(hidden_layer_qty)
    # #     end

    # #     it "input_size" do
    # #       rnn_simple.input_size.should eq(input_size)
    # #     end

    # #     it "output_size" do
    # #       rnn_simple.output_size.should eq(output_size)
    # #     end

    # #     it "hidden_size" do
    # #       rnn_simple.hidden_size.should eq(hidden_size_given)
    # #     end
    # #   end
    # # end

    # # context "when passing in invalid values (too small hidden_size_given)" do
    # #   # let(rnn_simple) { subject.new }
    # #   time_col_qty = -3
    # #   hidden_layer_qty = -2
    # #   input_size = -3
    # #   output_size = -2
    # #   hidden_size_given = -1

    # #   rnn_simple = Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
    # #     time_col_qty,
    # #     hidden_layer_qty,
    # #     input_size,
    # #     output_size,
    # #     hidden_size_given
    # #   )

    # #   it "has errors" do
    # #     rnn_simple.errors.empty?.should be_false
    # #     rnn_simple.errors.is_a?(Hash(Symbol, String)).should be_true
    # #     rnn_simple.errors.should eq({:time_col_qty => "time_col_qty must be at least 2!", :hidden_layer_qty => "hidden_layer_qty must be at least 1!", :input_size => "input_size must be at least 2", :output_size => "output_size must be at least 1", :hidden_size_given => "hidden_size_given must be at least 3 if supplied (otherwise it defaults to sum of @input_size and @output_size"})
    # #   end

    # #   it "is valid" do
    # #     rnn_simple.valid?.should be_false
    # #   end

    # #   context "has expected value for property" do
    # #     it "time_col_qty" do
    # #       rnn_simple.time_col_qty.should eq(time_col_qty)
    # #     end

    # #     it "hidden_layer_qty" do
    # #       rnn_simple.hidden_layer_qty.should eq(hidden_layer_qty)
    # #     end

    # #     it "input_size" do
    # #       rnn_simple.input_size.should eq(input_size)
    # #     end

    # #     it "output_size" do
    # #       rnn_simple.output_size.should eq(output_size)
    # #     end

    # #     it "hidden_size" do
    # #       rnn_simple.hidden_size.should eq(hidden_size_given)
    # #     end
    # #   end
    # # end
  end  
end



# context "as json" do
#   rnn_json = rnn_simple.to_json
#   # rnn_reformed = Ai4cr::NeuralNetwork::Cmn::RnnSimple.from_json(rnn_json)
#   it "rnn_json" do
#     rnn_json.should eq("")
#   end

#   # it "rnn_reformed" do
#   #   rnn_reformed.should eq("")
#   # end
  
#   context "has expected value for key" do
#     it "time_col_qty" do
#       rnn_json["time_col_qty"].should eq("") # 2)
#     end

#     it "hidden_layer_qty" do
#       rnn_json["hidden_layer_qty"].should eq("") # 1)
#     end

#     it "input_size" do
#       rnn_json["input_size"].should eq("") # 2)
#     end

#     it "output_size" do
#       rnn_json["output_size"].should eq("") # 1)
#     end

#     it "hidden_size" do
#       rnn_json["hidden_size"].should eq("") # 4)
#     end

#     it "errors" do
#       rnn_json["errors"].should eq(Hash(Symbol, String).new)
#       # rnn_simple.errors.should eq(Hash(Symbol, String).new)
#     end

#     it "valid?" do
#       rnn_json["valid?"].should be_true
#     end
#   end
# end