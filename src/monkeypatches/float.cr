struct Float64
  # These nets don't need huge numbers, so for sake a nets, we can set an artificial max (aka 'HIGH_ENOUGH_FOR_NETS'):
  HIGH_ENOUGH_FOR_NETS = 1.0e120 # 1.0e12 # Float64::MAX / (2**10) # MAX - 1

  # Float64.avoid_extremes(value)
  def self.avoid_extremes(value : Float64, alt_nan = Float64.new(0), alt_infin_pos = HIGH_ENOUGH_FOR_NETS, alt_infin_neg = -HIGH_ENOUGH_FOR_NETS)
    # For sake of ai4cr-internal calc's and for sake of to/from_json, we need to avoid NaN and Infinity.
    # In some cases, we want to specify alternate values
    case
    when value.nan?
      alt_nan
    when value > alt_infin_pos # value.infinite? && value > 0
      alt_infin_pos
    when value < alt_infin_neg # value.infinite? && value < 0
      alt_infin_neg
    else
      value
    end
  end

  def self.cap_extremes(value : Float64, alt_nan = Float64.new(0), alt_infin_pos = HIGH_ENOUGH_FOR_NETS, alt_infin_neg = -HIGH_ENOUGH_FOR_NETS)
    # For now, just re-use 'avoid_extremes' internally. Keep to distinguish or remove?
    avoid_extremes(value: value, alt_nan: alt_nan, alt_infin_pos: alt_infin_pos, alt_infin_neg: alt_infin_neg)
  end
end

# Looks like the smallest and biggest floats that work w/ to/from json are:
# * smallest: 1e-307 (1e-308 -> 0.0)
# * biggest : 1e+308
# So, to be safe, we need to cap our floats w/in those ranges.

# exp # => 307
# f = 10.0 ** (-exp) # => 9.999999999999995e-308
# f # => 9.999999999999995e-308
# j # => "9.999999999999995e-308"
# f2 # => 9.999999999999991e-308
# ********************
# exp # => 308
# f = 10.0 ** (-exp) # => 9.999999999999994e-309
# f # => 9.999999999999994e-309
# j # => "9.999999999999994e-309"
# f2 # => 0.0
# ********************
# exp # => 309
# f = 10.0 ** (-exp) # => 0.0
# f # => 0.0
# j # => "0.0"
# f2 # => 0.0

# ********************
# exp # => 308
# f = 10.0 ** (exp) # => 1.0000000000000006e+308
# f # => 1.0000000000000006e+308
# j # => "1.0000000000000006e+308"
# f2 # => 1.0000000000000012e+308
# ********************
# exp # => 309
# f = 10.0 ** (exp) # => Infinity
# f # => Infinity
# Unhandled exception: Infinity not allowed in JSON (JSON::Error)
#   from /usr/share/crystal/src/json/builder.cr:92:9 in 'number'
#   from /usr/share/crystal/src/json/to_json.cr:55:5 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:10:7 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:4:7 in 'to_json'
#   from examples/json_float.cr:13:7 in '__crystal_main'
#   from /usr/share/crystal/src/crystal/main.cr:110:5 in 'main_user_code'
#   from /usr/share/crystal/src/crystal/main.cr:96:7 in 'main'
#   from /usr/share/crystal/src/crystal/main.cr:119:3 in 'main'
#   from __libc_start_main
#   from _start
#   from ???
