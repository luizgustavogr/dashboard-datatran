const fs = require('fs');
const path = require('path');

const inputDirectory = __dirname;
const outputFileName = process.argv[2] || 'datatran_unificado.csv';
const outputPath = path.join(inputDirectory, outputFileName);

function findInputFiles(directory) {
	return fs
		.readdirSync(directory)
		.filter((fileName) => /^datatran\d{4}_processado\.csv$/i.test(fileName))
		.sort((left, right) => left.localeCompare(right, 'pt-BR'));
}

function splitCsvLine(line) {
	const values = [];
	let currentValue = '';
	let insideQuotes = false;

	for (let index = 0; index < line.length; index += 1) {
		const character = line[index];

		if (insideQuotes) {
			if (character === '"') {
				if (line[index + 1] === '"') {
					currentValue += '"';
					index += 1;
				} else {
					insideQuotes = false;
				}
			} else {
				currentValue += character;
			}
			continue;
		}

		if (character === ';') {
			values.push(currentValue);
			currentValue = '';
			continue;
		}

		if (character === '"' && currentValue === '') {
			insideQuotes = true;
			continue;
		}

		currentValue += character;
	}

	values.push(currentValue);
	return values;
}

function escapeCsvValue(value) {
	if (value === '') {
		return '';
	}

	const needsQuotes = /[;\n\r"]/.test(value);

	if (!needsQuotes) {
		return value;
	}

	return `"${value.replace(/"/g, '""')}"`;
}

function mergeCsvFiles(fileNames) {
	if (fileNames.length === 0) {
		throw new Error('Nenhum arquivo datatran_processado.csv foi encontrado.');
	}

	const orderedColumns = [];
	const seenColumns = new Set();
	const fileContents = [];

	for (const fileName of fileNames) {
		const filePath = path.join(inputDirectory, fileName);
		const content = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '');
		const lines = content.split(/\r?\n/).filter((line, index, array) => line.trim() !== '' || index < array.length - 1);

		if (lines.length === 0) {
			continue;
		}

		const currentHeader = lines[0];
		const dataLines = lines.slice(1);
		const columns = splitCsvLine(currentHeader);

		for (const column of columns) {
			if (!seenColumns.has(column)) {
				seenColumns.add(column);
				orderedColumns.push(column);
			}
		}

		fileContents.push({ columns, dataLines, fileName });
	}

	const mergedLines = [orderedColumns.join(';')];

	for (const { columns, dataLines, fileName } of fileContents) {
		const columnIndexByName = new Map(columns.map((column, index) => [column, index]));

		for (const line of dataLines) {
			if (line.trim() !== '') {
				const values = splitCsvLine(line);

				if (values.length !== columns.length) {
					throw new Error(`Quantidade de colunas inconsistente em ${fileName}.`);
				}

				const alignedValues = orderedColumns.map((column) => {
					const valueIndex = columnIndexByName.get(column);
					return escapeCsvValue(valueIndex === undefined ? '' : (values[valueIndex] ?? ''));
				});

				mergedLines.push(alignedValues.join(';'));
			}
		}
	}

	return mergedLines.join('\n') + '\n';
}

function main() {
	const inputFiles = findInputFiles(inputDirectory);
	const mergedCsv = mergeCsvFiles(inputFiles);

	fs.writeFileSync(outputPath, mergedCsv, 'utf8');
	console.log(`Arquivo criado com sucesso: ${outputPath}`);
	console.log(`Arquivos unificados: ${inputFiles.length}`);
}

try {
	main();
} catch (error) {
	console.error(error.message);
	process.exit(1);
}
