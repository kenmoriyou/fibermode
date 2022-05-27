{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          ipython
          python-lsp-server
          pyls-isort
        ]);

      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];
        };
      }
    );
}
