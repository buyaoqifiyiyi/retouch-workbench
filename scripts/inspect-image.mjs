import { existsSync } from 'node:fs';
import { mkdir, open, stat } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';

const DEFAULT_CLI = path.join(
  process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local'),
  'Programs',
  'ArtCraft',
  'PhotoCraft',
  'photocraft-cli.exe',
);
const CLI = process.env.PHOTOCRAFT_CLI || DEFAULT_CLI;

function usage() {
  console.error(
    'Usage:\n' +
    '  node inspect-image.mjs inspect <filePath>\n' +
    '  node inspect-image.mjs crop <inputPath> <outputPngPath> <x> <y> <width> <height>'
  );
  process.exit(2);
}

async function detectEncoding(filePath) {
  const handle = await open(filePath, 'r');
  const header = Buffer.alloc(12);
  try {
    await handle.read(header, 0, header.length, 0);
  } finally {
    await handle.close();
  }
  const hex = header.toString('hex');
  const ascii = header.toString('ascii');
  const ext = path.extname(filePath).toLowerCase();
  let format = null;
  let expectedExts = [];

  if (hex.startsWith('ffd8ff')) {
    format = 'JPEG';
    expectedExts = ['.jpg', '.jpeg'];
  } else if (hex.startsWith('89504e470d0a1a0a')) {
    format = 'PNG';
    expectedExts = ['.png'];
  } else if (ascii.startsWith('8BPS') && [1, 2].includes(header.readUInt16BE(4))) {
    format = header.readUInt16BE(4) === 2 ? 'PSB' : 'PSD';
    expectedExts = [format === 'PSB' ? '.psb' : '.psd'];
  } else if (hex.startsWith('49492a00') || hex.startsWith('4d4d002a')) {
    format = 'TIFF';
    expectedExts = ['.tif', '.tiff'];
  } else if (ascii.startsWith('BM')) {
    format = 'BMP';
    expectedExts = ['.bmp'];
  } else if (ascii.startsWith('RIFF') && ascii.slice(8, 12) === 'WEBP') {
    format = 'WEBP';
    expectedExts = ['.webp'];
  } else if (hex.startsWith('504b0304') && ext === '.pcraft') {
    format = 'PCRAFT';
    expectedExts = ['.pcraft'];
  }

  return {
    format,
    extension: ext,
    extension_matches: format ? expectedExts.includes(ext) : null,
  };
}

function runCli(args) {
  return execFileSync(CLI, args, { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
}

async function cmdInspect(targetPath) {
  const abs = path.resolve(targetPath);
  const infoStat = await stat(abs);
  const encoding = await detectEncoding(abs);
  const docJson = JSON.parse(runCli(['info', '--compact', abs]));
  console.log(JSON.stringify({
    path: abs,
    bytes: infoStat.size,
    ...encoding,
    document: docJson,
  }, null, 2));
}

async function cmdCrop(inputPath, outputPath, xRaw, yRaw, wRaw, hRaw) {
  const inAbs = path.resolve(inputPath);
  const outAbs = path.resolve(outputPath);
  if (inAbs === outAbs) {
    throw new Error('Refusing in-place crop: outputPngPath must differ from inputPath.');
  }
  if (existsSync(outAbs)) {
    throw new Error(`Refusing to overwrite existing file: ${outAbs}`);
  }
  const [x, y, width, height] = [xRaw, yRaw, wRaw, hRaw].map(Number);
  if (![x, y, width, height].every(Number.isFinite) || x < 0 || y < 0 || width <= 0 || height <= 0) {
    throw new Error('Invalid crop coordinates: expected x>=0, y>=0, width>0, height>0.');
  }

  await mkdir(path.dirname(outAbs), { recursive: true });
  const params = JSON.stringify({ x, y, width, height });
  runCli(['run', inAbs, '--cmd', 'image.crop', '--params', params, '--out', outAbs]);
  console.log(JSON.stringify({
    ok: true,
    input: inAbs,
    output: outAbs,
    crop: { x, y, width, height },
  }));
}

async function main() {
  const [, , sub, ...rest] = process.argv;
  if (sub === 'inspect' && rest.length === 1) {
    await cmdInspect(rest[0]);
  } else if (sub === 'crop' && rest.length === 6) {
    await cmdCrop(...rest);
  } else {
    usage();
  }
}

main().catch((err) => {
  console.error(err?.stderr ? `${err.message}\n${err.stderr}` : (err?.message || String(err)));
  process.exit(1);
});
