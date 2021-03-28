require "hardware"

memory = Hardware::Memory.new
p! memory.used         # => 2731404
p! memory.percent.to_i # => 32

cpu = Hardware::CPU.new
pid_stat = Hardware::PID.new.stat            # Default is Process.pid
app_stat = Hardware::PID.new("firefox").stat # Take the first matching PID

memory = Hardware::Memory.new
p! memory.used
cpu = Hardware::CPU.new
p! cpu.usage!

loop do
  sleep 1
  p! cpu.usage!.to_i          # => 17
  p! pid_stat.cpu_usage!      # => 1.5
  p! app_stat.cpu_usage!.to_i # => 4
end
