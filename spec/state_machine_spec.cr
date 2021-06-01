require "./spectator_helper"

# TODO: extract to a separate shard

Spectator.describe SimpleStateMachine::StateMachine do
  context "with all defaults used (i.e.: no params specified)" do
    let(state_machine) { SimpleStateMachine::StateMachine.new }

    describe "#initialize" do
      it "does not crash" do
        expect {
          SimpleStateMachine::StateMachine.new
        }.not_to raise_error
      end
    end

    describe "#next" do
      it "does not crash" do
        expect {
          state_machine.next
        }.not_to raise_error
      end
      it "By default, it returns the time and current_state" do
        state_before = state_machine.state_current
        p! state_before
        state_machine.next
        state_after = state_machine.state_current
        p! state_after
        expect(state_before).to eq(SimpleStateMachine::StateMachine::STATE_NEW)
        expect(state_after).to eq(SimpleStateMachine::StateMachine::STATE_INIT)
      end

      it "twice" do
        state_before = state_machine.state_current
        p! state_before
        state_machine.next
        state_mid = state_machine.state_current
        state_machine.next
        state_after = state_machine.state_current
        p! state_after
        expect(state_before).to eq(SimpleStateMachine::StateMachine::STATE_NEW)
        expect(state_mid).to eq(SimpleStateMachine::StateMachine::STATE_INIT)
        expect(state_after).to eq(SimpleStateMachine::StateMachine::STATE_READY)
      end
    end

    describe "#to_json" do
      let(obj_to_json_expected) {
        {
          "states": [
            "new",
            "init",
            "ready",
            "start",
            "running",
            "error",
            "stopped",
            "reset",
            "resetting",
            "success",
            "done",
          ],
          "state_current": "new",
          "history_size":  10,
          "history":       [
            "new",
          ],
          "allowed_paths": {
            "new": [
              "init",
              "error",
              "reset",
            ],
            "init": [
              "ready",
              "error",
              "reset",
            ],
            "ready": [
              "start",
              "error",
              "reset",
            ],
            "start": [
              "running",
              "error",
              "reset",
            ],
            "running": [
              "success",
              "error",
              "reset",
            ],
            "success": [
              "done",
              "error",
              "reset",
            ],
            "done": [
              "ready",
              "error",
              "reset",
            ],
            "error": [
              "stopped",
              "error",
              "reset",
            ],
            "stopped": [
              "reset",
              "error",
              "reset",
            ],
            "reset": [
              "resetting",
              "error",
              "reset",
            ],
            "resetting": [
              "ready",
              "error",
              "reset",
            ],
          },
          "use_default_paths": true,
        }.to_json
      }
      it "returns expected json" do
        expect(state_machine.to_json).to eq(obj_to_json_expected)
      end
    end
  end

  context "with custom states specified" do
    let(state0) { :off }
    let(state1) { :red }
    let(state2) { :green }
    let(state3) { :blue }
    let(states) {
      [state0, state1, state2, state3]
    }

    let(state_machine) { SimpleStateMachine::StateMachine.new(states: states) }

    describe "#initialize" do
      it "does not crash" do
        expect {
          SimpleStateMachine::StateMachine.new
        }.not_to raise_error
      end
    end

    describe "#next" do
      it "does not crash" do
        expect {
          state_machine.next
        }.not_to raise_error
      end
      it "By default, it returns the time and current_state" do
        # expect(Time).to receive(:utc)
        # expect(state_machine).to receive(:goto)
        state_before = state_machine.state_current
        p! state_before
        state_machine.next
        state_after = state_machine.state_current
        p! state_after
        expect(state_before).to eq(state0)
        expect(state_after).to eq(state1)
      end
    end

    describe "#to_json" do
      let(obj_to_json_expected) {
        {
          "states": [
            state0,
            state1,
            state2,
            state3,
          ],
          "state_current": state0,
          "history_size":  10,
          "history":       [
            state0,
          ],
          "allowed_paths": {
            state0 => [
              state1,
              state2,
              state3,
            ],
            state1 => [
              state0,
              state2,
              state3,
            ],
            state2 => [
              state0,
              state1,
              state3,
            ],
            state3 => [
              state0,
              state1,
              state2,
            ],
          },
          "use_default_paths": true,
        }.to_json
      }
      it "returns expected json" do
        expect(state_machine.to_json).to eq(obj_to_json_expected)
      end
    end
  end
end
