{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    poetry2nix,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
      };
      inherit (pkgs) python311;
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        inherit python311;
        projectDir = ./.;
        preferWheels = true;
        # overrides = poetry2nix.overrides.withDefaults overrides;
      };
      my-typst-preview = import ./pkgs/typst-preview/package.nix {
        inherit (pkgs) lib rustPlatform fetchFromGitHub mkYarnPackage fetchYarnDeps pkg-config libgit2 openssl zlib stdenv darwin;
      };

      buildInputs = with pkgs; [
        imagemagick
        my-typst-preview
        pandoc
        pdf2svg
        typst
        typstfmt
        typst-lsp
        wget
        # poetryEnv
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
