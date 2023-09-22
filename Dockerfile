# Engine
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2

# Bindings
pip install stockfish 
brew install stockfish