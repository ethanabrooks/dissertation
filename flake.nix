{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryEnv;
      python = pkgs.python311;
      poetryEnv = mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
      };

      buildInputs = with pkgs; [
        poetryEnv
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
