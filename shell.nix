{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [ fmt gsl zlib freetype ffmpeg libconfig lmdb libtiff eigen gtk4];
  inputsFrom = with pkgs; [ fmt gsl zlib freetype ffmpeg libconfig lmdb libtiff eigen gtk4];
  hardeningDisable = ["all"];
}
