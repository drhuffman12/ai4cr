class SafeCounter
  # Based on and with much thanks to: https://itnext.io/comparing-crystals-concurrency-with-that-of-go-part-ii-89049701b1a5
  #
  # Example Usage:
  # ```
  # c = SafeCounter.new
  # (1..1000).each {
  #   spawn c.inc("someKey")
  # }
  #
  # sleep 1.second
  # puts c.value("someKey")
  # ```

  @@v = Hash(String, Int32).new(0)

  def initialize
    @mux = Mutex.new
  end

  def value(key : String)
    @mux.lock
    @@v[key]
  ensure
    @mux.unlock
  end

  def inc(key : String)
    @mux.lock
    val = @@v.has_key?(key) ? @@v[key] + 1 : 1
    @@v[key] = val
  ensure
    @mux.unlock
    val
  end

  def reset(key : String, val : Int32)
    # Unfortunately, mutex's don't seem to work with to_/from_json, so we'll use a 'reset' method.
    # i.e.: include JSON::Serializable => Error: no overload matches 'Mutex#to_json' with type JSON::Builder
    @mux.lock
    @@v[key] = val
  ensure
    @mux.unlock
    val
  end
end
